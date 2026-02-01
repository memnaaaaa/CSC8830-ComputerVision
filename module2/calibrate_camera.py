"""
Camera Calibration Script - Step 1 (Fixed Version)
Key corrections marked with # FIXED:
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path


def calibrate_camera(image_dir, visualize=False):
    """
    Calibrate using 7x9 inner corners (8x10 squares), 20mm square size
    """
    # Explicitly defined these as inner corners (not squares)
    # Pattern has 8x10 squares, which gives 7x9 inner intersections
    CHECKERBOARD_ROWS = 7  # Vertical inner corners
    CHECKERBOARD_COLS = 9  # Horizontal inner corners
    SQUARE_SIZE_MM = 20.0
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Object point generation 
    # Create grid of object points. Shape is (rows*cols, 3)
    # Columns = width (x-direction), Rows = height (y-direction)
    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    
    # np.mgrid order (x:cols, y:rows) i.e., (9, 7) not (7, 9)
    # Creates coordinates: (0,0), (1,0), ... (8,0), (0,1), (1,1)...
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM  # Scale to mm (20mm per square)
    
    objpoints = []
    imgpoints = []
    image_size = None  # Track this to enforce consistency
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                        glob.glob(os.path.join(image_dir, "*.png")) +
                        glob.glob(os.path.join(image_dir, "*.jpeg")))
    
    print(f"Processing {len(image_paths)} images...")
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Enforce consistent image size
        # Previous failures occurred because iPhone images had mixed resolutions
        # or orientations (some portrait 4284x5712, some landscape 5712x4284),
        # causing the optimization to diverge with 29k focal lengths.
        if image_size is None:
            image_size = (w, h)
            print(f"Reference resolution: {w}x{h}")
        elif (w, h) != image_size:
            # Skip images with different resolution/orientation
            print(f"WARNING: {Path(img_path).name} size {w}x{h} != {image_size}, skipping!")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # OpenCV wants (cols, rows) = (9, 7), not (7, 9)
        # findChessboardCorners API expects (width, height) i.e., (9, 7)
        ret, corners = cv2.findChessboardCorners(
            gray, 
            (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),  # (9, 7)
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            
            if visualize:
                # Use (cols, rows) here too for visualization
                cv2.drawChessboardCorners(img, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), 
                                         corners_refined, ret)
                # Resize huge images for display (iPhone photos are massive)
                display = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
                cv2.imshow('Corners', display)
                cv2.waitKey(100)
        else:
            print(f"Failed to find corners: {Path(img_path).name}")
    
    cv2.destroyAllWindows()
    
    if len(objpoints) < 5:
        raise ValueError(f"Only {len(objpoints)} valid images! Need at least 5.")
    
    print(f"Calibrating with {len(objpoints)} images at resolution {image_size}...")
    
    # Perform calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    # Calculate RMS reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    return ret, K, dist, rvecs, tvecs, mean_error, objpoints, imgpoints


if __name__ == "__main__":
    try:
        ret, K, dist, rvecs, tvecs, rms, objpts, imgpts = calibrate_camera(
            "data/calibration_images", 
            visualize=True
        )
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Camera Matrix K:\n{K}")
        print(f"\nFocal Lengths: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        print(f"Principal Point: ({K[0,2]:.1f}, {K[1,2]:.1f})")
        
        # Validate aspect ratio (should be ~1.0 for square pixels)
        aspect_ratio = K[0,0]/K[1,1]
        print(f"Aspect Ratio fx/fy: {aspect_ratio:.3f}")
        
        print(f"\nDistortion: k1={dist[0,0]:.4f}, k2={dist[0,1]:.4f}")
        print(f"RMS Reprojection Error: {rms:.3f} pixels")
        
        # Success criteria check
        if 0.9 < aspect_ratio < 1.1 and rms < 1.0:
            print("\n✓ CALIBRATION SUCCESSFUL")
            np.savez("camera_params.npz", K=K, dist=dist)
            print("✓ Saved to camera_params.npz")
        else:
            print("\n✗ CALIBRATION FAILED")
            
    except Exception as e:
        print(f"Error: {e}")