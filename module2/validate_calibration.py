"""
module2/validate_calibration.py
Step 3: Validation Experiment (>2 meters distance)
Validates accuracy of homography-based measurement against ground truth.
Using camera calibration from Step 1 and reference object (A4 paper)
"""

import cv2 # OpenCV for computer vision tasks
import numpy as np # NumPy for numerical operations
import math # For mathematical functions


def compute_homography(image_pts, world_pts):
    """Compute H and H_inv."""
    img_pts = np.array(image_pts, dtype=np.float32)
    wrld_pts = np.array(world_pts, dtype=np.float32)
    H = cv2.getPerspectiveTransform(wrld_pts, img_pts)
    return H, np.linalg.inv(H)


def measure_distance(pt1_img, pt2_img, H_inv):
    """Calculate distance in mm."""
    p1 = np.array([pt1_img[0], pt1_img[1], 1.0])
    p2 = np.array([pt2_img[0], pt2_img[1], 1.0])
    
    w1 = H_inv.dot(p1)
    w2 = H_inv.dot(p2)
    
    X1, Y1 = w1[0]/w1[2], w1[1]/w1[2]
    X2, Y2 = w2[0]/w2[2], w2[1]/w2[2]
    
    return math.sqrt((X2-X1)**2 + (Y2-Y1)**2)


def main():
    print("Step 3: Validation (>2m) - No Undistortion")
    print("="*60)
    
    # Ground truth measurements
    camera_distance_m = 2.30  # Measured distance
    ground_truth_mm = 75.0   # Tape measure of object
    
    # Image coords (from get_coordinates.py)
    # REPLACE with actual detected points
    ref_points = [
        [1857, 2829],   # A4 TL
        [2245, 2828],  # A4 TR  
        [2245, 3331],  # A4 BR
        [1856, 3331]    # A4 BL
    ]
    
    # REPLACE with actual detected object points
    obj_pt1 = [2335, 2997]
    obj_pt2 = [2332, 3128]
    
    # World coords (A4: 210x297 mm)
    world_points = [[0,0], [210,0], [210,297], [0,297]]
    
    # Load image (no undistortion)
    img = cv2.imread("data/validation_image.jpg")
    if img is None:
        print("Error: validation_image.jpg not found")
        return
    
    # Compute homography
    H, H_inv = compute_homography(ref_points, world_points)
    
    # Measure
    measured = measure_distance(obj_pt1, obj_pt2, H_inv)
    
    # Error analysis
    abs_error = abs(measured - ground_truth_mm)
    pct_error = (abs_error / ground_truth_mm) * 100
    
    print(f"Camera distance: {camera_distance_m} m (>2m: {'Yes' if camera_distance_m > 2 else 'No'})")
    print(f"Ground truth:    {ground_truth_mm:.1f} mm")
    print(f"Measured:        {measured:.1f} mm")
    print(f"Absolute error:  {abs_error:.2f} mm")
    print(f"Percent error:   {pct_error:.2f}%")
    print(f"Status:          {'PASS' if pct_error < 5 else 'FAIL'} (<5% threshold)")
    print("="*60)
    
    # Visualize
    for pt in ref_points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
    cv2.circle(img, (int(obj_pt1[0]), int(obj_pt1[1])), 8, (0, 0, 255), -1)
    cv2.circle(img, (int(obj_pt2[0]), int(obj_pt2[1])), 8, (0, 0, 255), -1)
    cv2.line(img, (int(obj_pt1[0]), int(obj_pt1[1])), 
             (int(obj_pt2[0]), int(obj_pt2[1])), (255, 0, 0), 2)
    
    cv2.imshow(f"Error: {pct_error:.1f}%", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
