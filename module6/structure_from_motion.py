"""
module6/structure_from_motion.py
Complete Structure from Motion - Planar Object (4 Views)
Reconstructs 3D structure of a planar object and estimates its boundary
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


class PlanarSfM:
    def __init__(self, K):
        """
        Initialize with camera intrinsics.
        
        Args:
            K: 3x3 intrinsic matrix from calibration
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.images = []
        self.keypoints_list = []
        self.descriptors_list = []
        self.H_list = []  # Homographies relative to view 0
        self.camera_poses = []  # (R, t) for each view
        self.points_3d = None
        
    def load_images(self, image_paths):
        """Load 4 images."""
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Cannot load: {path}")
            self.images.append(img)
            print(f"Loaded: {path}, shape: {img.shape}")
        
    def detect_features(self):
        """Detect SIFT features in all images."""
        sift = cv2.SIFT_create(nfeatures=2000)
        
        for i, img in enumerate(self.images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            self.keypoints_list.append(kp)
            self.descriptors_list.append(des)
            print(f"  View {i}: {len(kp)} features detected")
    
    def match_features(self, idx1, idx2, ratio_thresh=0.75):
        """
        Match features between two views using Lowe's ratio test.
        """
        if self.descriptors_list[idx1] is None or self.descriptors_list[idx2] is None:
            return None, None, []
        
        # FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(self.descriptors_list[idx1], 
                                self.descriptors_list[idx2], k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            print(f"  Warning: Only {len(good_matches)} good matches")
            return None, None, []
        
        # Extract points
        pts1 = np.float32([self.keypoints_list[idx1][m.queryIdx].pt 
                          for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([self.keypoints_list[idx2][m.trainIdx].pt 
                          for m in good_matches]).reshape(-1, 1, 2)
        
        return pts1, pts2, good_matches
    
    def estimate_homography(self, idx1, idx2, threshold=3.0):
        """
        Estimate homography H between two views using RANSAC.
        """
        pts1, pts2, matches = self.match_features(idx1, idx2)
        
        if pts1 is None or len(pts1) < 4:
            print(f"  Error: Not enough matches for homography")
            return None, None, None
        
        # RANSAC to find H
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
        
        if H is None:
            print(f"  Error: Homography estimation failed")
            return None, None, None
        
        inliers = int(np.sum(mask))
        print(f"  Homography {idx1}->{idx2}: {inliers}/{len(matches)} inliers ({100*inliers/len(matches):.1f}%)")
        
        # Keep only inliers
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]
        
        # Visualize
        self._visualize_matches(idx1, idx2, matches, mask, f"matches_{idx1}_{idx2}.png")
        
        return H, pts1_inliers, pts2_inliers
    
    def decompose_homography(self, H):
        """
        Decompose H into rotation R, translation t, and plane normal n.
        """
        # Normalize H
        H_prime = self.K_inv @ H @ self.K
        
        # Extract columns
        h1 = H_prime[:, 0]
        h2 = H_prime[:, 1]
        h3 = H_prime[:, 2]
        
        # Compute scale
        scale1 = np.linalg.norm(h1)
        scale2 = np.linalg.norm(h2)
        scale = (scale1 + scale2) / 2.0
        
        if scale < 1e-10:
            scale = 1.0
        
        # Rotation columns
        if scale1 < 1e-10:
            scale1 = 1.0
        if scale2 < 1e-10:
            scale2 = 1.0
        r1 = h1 / scale1
        r2 = h2 / scale2
        r3 = np.cross(r1, r2)
        
        # Ensure right-handed coordinate system
        if np.linalg.det(np.column_stack([r1, r2, r3])) < 0:
            r3 = -r3
        
        R = np.column_stack([r1, r2, r3])
        
        # Ensure rotation matrix is valid (orthogonal)
        U, S, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # Translation - ensure 1D array
        t = (h3 / scale).flatten()
        
        # Plane normal in world coordinates (Z=0 plane)
        n = np.array([0, 0, 1.0])
        
        # Four solutions from ambiguity in sign
        solutions = []
        
        # Solution 1
        solutions.append((R, t, n))
        
        # Solution 2 (alternative interpretation)
        R_flip = R.copy()
        R_flip[:, 2] = -R_flip[:, 2]
        solutions.append((R_flip, -t, n))
        
        return solutions
    
    def _select_best_solution(self, solutions, idx):
        """
        Select best solution based on cheirality (points in front of camera).
        """
        best_solution = solutions[0]
        best_score = -1
        
        for R, t, n in solutions:
            # Ensure t is at least 1D array
            t = np.asarray(t).flatten()
            
            # Camera center C = -R^T * t
            C = -R.T @ t
            
            # Ensure C is array, not scalar
            C = np.asarray(C).flatten()
            
            # Check if camera has valid position
            if len(C) < 3:
                continue
                
            # Score: prefer camera with positive Z and reasonable distance
            score = C[2] if C[2] > 0 else 0
            
            if score > best_score:
                best_score = score
                best_solution = (R, t, n)
        
        return best_solution
    
    def triangulate_planar_point(self, pts_img, camera_poses):
        """
        Triangulate a 3D point on plane Z=0 from multiple views.
        """
        X_points = []
        
        for i in range(len(pts_img) - 1):
            x1 = np.array([pts_img[i][0], pts_img[i][1], 1.0])
            x2 = np.array([pts_img[i+1][0], pts_img[i+1][1], 1.0])
            
            R1, t1 = camera_poses[i]
            R2, t2 = camera_poses[i+1]
            
            # Ensure t is 1D array
            if t1.ndim > 1:
                t1 = t1.flatten()
            if t2.ndim > 1:
                t2 = t2.flatten()
            
            # Camera centers
            C1 = -R1.T @ t1
            C2 = -R2.T @ t2
            
            # Ray directions
            d1 = R1.T @ self.K_inv @ x1
            d2 = R2.T @ self.K_inv @ x2
            
            # Intersection with Z=0 plane
            if abs(d1[2]) < 1e-6 or abs(d2[2]) < 1e-6:
                continue
            
            lambda1 = -C1[2] / d1[2]
            lambda2 = -C2[2] / d2[2]
            
            X1 = C1 + lambda1 * d1
            X2 = C2 + lambda2 * d2
            
            # Ensure both are 1D arrays of shape (3,)
            X1 = np.asarray(X1).flatten()
            X2 = np.asarray(X2).flatten()
            
            # Average the two estimates
            X = (X1 + X2) / 2
            X_points.append(X)
        
        if not X_points:
            return None
        
        # Stack into array and take mean
        X_points = np.array(X_points)
        return np.mean(X_points, axis=0)
    
    def _triangulate_all_points(self):
        """Triangulate all matched points across views."""
        print("  Triangulating 3D points...")
        
        # Use points from view 0-1 matching as base
        _, pts1, pts2 = self.estimate_homography(0, 1)
        
        if pts1 is None:
            # Try other pairs
            for i in range(1, 4):
                _, pts1, pts2 = self.estimate_homography(0, i)
                if pts1 is not None:
                    break
        
        if pts1 is None:
            raise ValueError("Could not find valid homography for any pair")
        
        points_3d = []
        
        # For each matched point, find its location in other views and triangulate
        for i in range(min(len(pts1), 100)):  # Limit for speed
            # Get point in view 0
            p0 = pts1[i].ravel()
            
            # Project to other views using homographies
            p_homo = np.array([p0[0], p0[1], 1.0])
            
            pts_all_views = [p0]
            for H in self.H_list[1:]:
                p_proj = H @ p_homo
                p_proj = p_proj[:2] / p_proj[2]
                pts_all_views.append(p_proj)
            
            # Triangulate
            X = self.triangulate_planar_point(pts_all_views, self.camera_poses)
            if X is not None:
                points_3d.append(X)
        
        if not points_3d:
            raise ValueError("Triangulation produced no valid 3D points")
        self.points_3d = np.array(points_3d).reshape(-1, 3)
        print(f"  Triangulated {len(points_3d)} 3D points")
        
        return self.points_3d
    
    def _estimate_boundary(self, points_3d):
        """
        Estimate object boundary using convex hull.
        """
        if len(points_3d) < 3:
            print("  Not enough points for boundary estimation")
            return None
        
        # Project to XY plane (since Z≈0 for planar object)
        points_2d = points_3d[:, :2]
        
        # Convex hull
        try:
            hull = ConvexHull(points_2d)
            boundary_points = points_3d[hull.vertices]
            
            # Close the loop
            boundary_points = np.vstack([boundary_points, boundary_points[0]])
            
            print(f"  Boundary: {len(hull.vertices)} vertices")
            return boundary_points
        except Exception as e:
            print(f"  Convex hull failed: {e}")
            return None
    
    def _visualize_matches(self, idx1, idx2, matches, mask, filename):
        """Save match visualization."""
        img_matches = cv2.drawMatches(
            self.images[idx1], self.keypoints_list[idx1],
            self.images[idx2], self.keypoints_list[idx2],
            [m for i, m in enumerate(matches) if mask.ravel()[i]],
            None, 
            matchColor=(0, 255, 0),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(filename, img_matches)
        print(f"    Saved: {filename}")
    
    def reconstruct(self):
        """Full SfM pipeline."""
        print("\n" + "="*70)
        print("STRUCTURE FROM MOTION - PLANAR OBJECT")
        print("="*70)
        
        # Step 1: Feature detection
        print("\nStep 1: Feature Detection")
        self.detect_features()
        
        # Step 2: Estimate homographies (view 0 as reference)
        print("\nStep 2: Homography Estimation")
        self.H_list = [np.eye(3)]  # H_00 = I
        
        for i in range(1, 4):
            H, _, _ = self.estimate_homography(0, i)
            if H is not None:
                self.H_list.append(H)
            else:
                print(f"  Warning: Using identity for view {i}")
                self.H_list.append(np.eye(3))
        
        # Step 3: Decompose homographies to poses
        print("\nStep 3: Camera Pose Recovery (Homography Decomposition)")
        self.camera_poses = []  # Reset poses
        self.camera_poses.append((np.eye(3), np.zeros(3)))  # View 0 at origin
        
        for i, H in enumerate(self.H_list[1:], 1):
            solutions = self.decompose_homography(H)
            R, t, n = self._select_best_solution(solutions, i)

            # Ensure t is 1D array
            t = np.asarray(t).flatten()

            self.camera_poses.append((R, t.flatten()))
            
            C = -R.T @ t  # Camera center
            print(f"  View {i}:")
            print(f"    Rotation:\n{R}")
            print(f"    Translation: {t.flatten()}")
            print(f"    Camera center: {C.flatten()}")
            print(f"    Distance from origin: {np.linalg.norm(C):.2f} units")
        
        # Step 4: Triangulate 3D points
        print("\nStep 4: Triangulation")
        points_3d = self._triangulate_all_points()
        
        # Step 5: Boundary estimation
        print("\nStep 5: Boundary Estimation")
        boundary = self._estimate_boundary(points_3d)
        
        # Step 6: Save reconstruction info
        self._save_reconstruction_info(points_3d, boundary)
        
        return points_3d, boundary
    
    def _save_reconstruction_info(self, points_3d, boundary):
        """Save numerical results for report."""
        with open("reconstruction_results.txt", "w") as f:
            f.write("STRUCTURE FROM MOTION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write("Camera Intrinsics (K):\n")
            f.write(str(self.K) + "\n\n")
            
            f.write("Camera Poses (R, t):\n")
            for i, (R, t) in enumerate(self.camera_poses):
                f.write(f"View {i}:\n")
                f.write(f"  R = \n{R}\n")
                f.write(f"  t = {t.flatten()}\n")
                C = -R.T @ t
                f.write(f"  C = {C.flatten()} (camera center)\n\n")
            
            f.write(f"Reconstructed {len(points_3d)} 3D points\n")
            f.write(f"Point cloud extent:\n")
            f.write(f"  X: [{points_3d[:,0].min():.3f}, {points_3d[:,0].max():.3f}]\n")
            f.write(f"  Y: [{points_3d[:,1].min():.3f}, {points_3d[:,1].max():.3f}]\n")
            f.write(f"  Z: [{points_3d[:,2].min():.3f}, {points_3d[:,2].max():.3f}]\n")
            
            if boundary is not None:
                f.write(f"\nBoundary has {len(boundary)-1} vertices\n")
        
        print("  Saved: reconstruction_results.txt")
    
    def visualize_3d(self, points_3d, boundary=None):
        """Plot 3D reconstruction."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c='blue', marker='.', s=10, alpha=0.6, label='Reconstructed points')
        
        # Plot camera positions and viewing directions
        for i, (R, t) in enumerate(self.camera_poses):
            C = -R.T @ t  # Camera center
            ax.scatter(C[0], C[1], C[2], c='red', marker='^', s=200, edgecolors='black')
            ax.text(C[0], C[1], C[2], f'  Cam{i}', fontsize=10, color='red')
            
            # Viewing direction (Z-axis of camera)
            direction = R[2, :]  # Third row of R is optical axis
            ax.quiver(C[0], C[1], C[2], 
                     direction[0]*0.3, direction[1]*0.3, direction[2]*0.3,
                     color='red', arrow_length_ratio=0.3, alpha=0.5)
        
        # Plot boundary
        if boundary is not None:
            ax.plot(boundary[:, 0], boundary[:, 1], boundary[:, 2],
                   'g-', linewidth=3, label='Estimated boundary')
            ax.scatter(boundary[:-1, 0], boundary[:-1, 1], boundary[:-1, 2],
                      c='green', marker='o', s=50, edgecolors='black')
        
        # Reference plane Z=0
        xx, yy = np.meshgrid(
            np.linspace(points_3d[:,0].min(), points_3d[:,0].max(), 10),
            np.linspace(points_3d[:,1].min(), points_3d[:,1].max(), 10)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        
        # Set labels and title
        ax.set_xlabel('X (units)')
        ax.set_ylabel('Y (units)')
        ax.set_zlabel('Z (units)')
        ax.set_title('Structure from Motion: Planar Object Reconstruction')
        
        ax.legend(loc='upper left', fontsize=8)
        
        # Equal aspect ratio
        max_range = np.array([
            points_3d[:,0].max() - points_3d[:,0].min(),
            points_3d[:,1].max() - points_3d[:,1].min(),
            points_3d[:,2].max() - points_3d[:,2].min()
        ]).max() / 2.0
        
        mid_x = (points_3d[:,0].max() + points_3d[:,0].min()) * 0.5
        mid_y = (points_3d[:,1].max() + points_3d[:,1].min()) * 0.5
        mid_z = (points_3d[:,2].max() + points_3d[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.savefig('sfm_reconstruction_3d.png', dpi=150, bbox_inches='tight')
        print("Saved: sfm_reconstruction_3d.png")
        plt.show()


def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # Camera calibration from Module 2:
    K = np.array([
        [4237.1, 0, 2189.7],    # fx, 0, cx (from iPhone calibration)
        [0, 4260.4, 2685.9],    # 0, fy, cy
        [0, 0, 1]               # 0, 0, 1
    ])
    
    # Or use approximate values if recalibrating:
    # K = np.array([
    #     [1000, 0, 640],
    #     [0, 1000, 360],
    #     [0, 0, 1]
    # ])
    
    # 4 images of a book on floor (different viewpoints)
    image_paths = [
        "data/book_view1.jpg",   # Front/central view (reference)
        "data/book_view2.jpg",   # ~30cm to the right
        "data/book_view3.jpg",   # ~30cm to the left  
        "data/book_view4.jpg"    # From above/angle
    ]
    
    # Camera position descriptions for report
    camera_positions = [
        "View 0 (Reference): Camera perpendicular to book, 1m height",
        "View 1: Camera moved ~30cm right",
        "View 2: Camera moved ~30cm left", 
        "View 3: Camera moved forward, ~15° tilt down"
    ]
    
    print("Camera Setup Description (for report):")
    for desc in camera_positions:
        print(f"  {desc}")
    
    # ============================================================
    # RUN SFM PIPELINE
    # ============================================================
    
    sfm = PlanarSfM(K)
    sfm.load_images(image_paths)
    points_3d, boundary = sfm.reconstruct()
    sfm.visualize_3d(points_3d, boundary)
    
    # Summary
    print("\n" + "="*70)
    print("RECONSTRUCTION SUMMARY")
    print("="*70)
    print(f"Total 3D points reconstructed: {len(points_3d)}")
    print(f"Object dimensions (approximate):")
    print(f"  Width (X):  {points_3d[:,0].max() - points_3d[:,0].min():.2f} units")
    print(f"  Height (Y): {points_3d[:,1].max() - points_3d[:,1].min():.2f} units")
    print(f"  Planarity (Z std): {np.std(points_3d[:,2]):.4f} units")
    print(f"\nOutput files generated:")
    print(f"  - matches_*.png (feature correspondences)")
    print(f"  - sfm_reconstruction_3d.png (3D visualization)")
    print(f"  - reconstruction_results.txt (numerical data)")
    print("="*70)


if __name__ == "__main__":
    main()
