"""
module2/measure_object.py
Step 2: Object Measurement via Homography
Using camera calibration from Step 1 and reference object (A4 paper)
"""

import cv2 # OpenCV for computer vision tasks
import numpy as np # NumPy for numerical operations
import math # For mathematical functions


def compute_homography(image_pts, world_pts):
    """
    Compute Homography H mapping world coordinates (X,Y) to image (u,v).
    Math: s * [u, v, 1]^T = H * [X, Y, 1]^T
    """
    img_pts = np.array(image_pts, dtype=np.float32)
    wrld_pts = np.array(world_pts, dtype=np.float32)
    H = cv2.getPerspectiveTransform(wrld_pts, img_pts)
    return H, np.linalg.inv(H)


def image_to_world(image_pt, H_inv):
    """Map image point (u,v) to world coordinates (X,Y,mm)."""
    p = np.array([image_pt[0], image_pt[1], 1.0])
    world_h = H_inv.dot(p)
    X = world_h[0] / world_h[2]
    Y = world_h[1] / world_h[2]
    return X, Y


def measure_distance(pt1_img, pt2_img, H_inv):
    """Calculate Euclidean distance between two image points."""
    X1, Y1 = image_to_world(pt1_img, H_inv)
    X2, Y2 = image_to_world(pt2_img, H_inv)
    return math.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)


def main():
    # Configuration - Update these coordinates
    image_file = "data/object_to_measure.jpg"
    
    # Reference: A4 paper corners in image (pixels) - Order: TL, TR, BR, BL
    ref_image_points = [
        [928, 1704],   # Top-left
        [2638, 1694],  # Top-right
        [2621, 3947],  # Bottom-right
        [893, 3906]    # Bottom-left
    ]
    
    # Real world: A4 dimensions in mm
    ref_world_points = [
        [0, 0],       # Top-left
        [210, 0],     # Top-right (A4 width)
        [210, 297],   # Bottom-right (A4 height)
        [0, 297]      # Bottom-left
    ]
    
    # Object to measure
    object_pt1 = [3387, 2296]
    object_pt2 = [3353, 3675]
    
    # Load image (original, no undistortion)
    img = cv2.imread(image_file)
    if img is None:
        print(f"Error: Cannot load {image_file}")
        return
    
    # Compute homography directly on original image
    H, H_inv = compute_homography(ref_image_points, ref_world_points)
    
    # Measure
    distance_mm = measure_distance(object_pt1, object_pt2, H_inv)
    
    # Results
    print("="*50)
    print("MEASUREMENT RESULT")
    print("="*50)
    w1 = image_to_world(object_pt1, H_inv)
    w2 = image_to_world(object_pt2, H_inv)
    print(f"World: ({w1[0]:.1f}, {w1[1]:.1f}) to ({w2[0]:.1f}, {w2[1]:.1f}) mm")
    print(f"Distance: {distance_mm:.2f} mm ({distance_mm/10:.2f} cm)")
    print("="*50)
    
    # Visualize
    for pt in ref_image_points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
    cv2.circle(img, (int(object_pt1[0]), int(object_pt1[1])), 8, (0, 0, 255), -1)
    cv2.circle(img, (int(object_pt2[0]), int(object_pt2[1])), 8, (0, 0, 255), -1)
    cv2.line(img, (int(object_pt1[0]), int(object_pt1[1])), 
             (int(object_pt2[0]), int(object_pt2[1])), (255, 0, 0), 2)
    
    cv2.imshow("Measurement", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
