"""
module6/motion_tracking_validation.py
Part B: Motion Tracking from Fundamentals + Bilinear Interpolation
Derives tracking equations and validates with actual pixel locations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def theoretical_background():
    """
    Prints the theoretical derivation for the report.
    """
    theory = """
    ============================================================================
    THEORETICAL DERIVATION: MOTION TRACKING & BILINEAR INTERPOLATION
    ============================================================================
    
    1. BRIGHTNESS CONSTANCY ASSUMPTION
    ----------------------------------
    The fundamental assumption of optical flow:
    
        I(x, y, t) = I(x + u, y + v, t + 1)
    
    Where (u, v) is the motion vector (optical flow).
    
    Taylor series expansion (first order):
        I(x+u, y+v, t+1) ≈ I(x,y,t) + ∂I/∂x · u + ∂I/∂y · v + ∂I/∂t · 1
    
    Setting equal and rearranging:
        ∂I/∂x · u + ∂I/∂y · v + ∂I/∂t = 0
    
    Or: Ix·u + Iy·v + It = 0  (Single equation, two unknowns → aperture problem)
    
    2. LUCAS-KANADE METHOD (Local Solution)
    ----------------------------------------
    Assume constant flow in local window. Solve least squares:
    
    [ ΣIx²   ΣIxIy ] [ u ] = [ -ΣIxIt ]
    [ ΣIxIy  ΣIy²  ] [ v ]   [ -ΣIyIt ]
    
    A · d = b  →  d = (AᵀA)⁻¹Aᵀb  (if AᵀA invertible)
    
    3. BILINEAR INTERPOLATION
    -------------------------
    For sub-pixel tracking, we need intensity at non-integer locations.
    
    Given four corners Q11=(x1,y1), Q12=(x1,y2), Q21=(x2,y1), Q22=(x2,y2)
    with intensities I(Q11), I(Q12), I(Q21), I(Q22):
    
    First interpolate in x:
        I(x,y1) = I(Q11)·(x2-x)/(x2-x1) + I(Q21)·(x-x1)/(x2-x1)
        I(x,y2) = I(Q12)·(x2-x)/(x2-x1) + I(Q22)·(x-x1)/(x2-x1)
    
    Then interpolate in y:
        I(x,y) = I(x,y1)·(y2-y)/(y2-y1) + I(x,y2)·(y-y1)/(y2-y1)
    
    Simplified (unit square, x1=y1=0, x2=y2=1):
        I(x,y) = I(0,0)(1-x)(1-y) + I(1,0)x(1-y) + I(0,1)(1-x)y + I(1,1)xy
    
    4. TRACKING EQUATION
    -------------------
    For frame-to-frame tracking with sub-pixel accuracy:
    
        p(t+1) = p(t) + v(p,t)  where v is flow vector
    
    Using bilinear interpolation to get v at sub-pixel location p.
    ============================================================================
    """
    print(theory)
    return theory


def bilinear_interpolate(image, x, y):
    """
    Compute bilinear interpolation at sub-pixel location (x, y).
    
    Args:
        image: 2D numpy array
        x, y: floating point coordinates
    
    Returns:
        Interpolated intensity value
    """
    h, w = image.shape
    
    # Clip to valid range
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    
    # Four corner coordinates
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
    
    # Fractional parts
    dx, dy = x - x1, y - y1
    
    # Corner intensities
    Q11 = image[y1, x1]
    Q12 = image[y1, x2]
    Q21 = image[y2, x1]
    Q22 = image[y2, x2]
    
    # Bilinear interpolation
    I = (Q11 * (1 - dx) * (1 - dy) +
         Q12 * dx * (1 - dy) +
         Q21 * (1 - dx) * dy +
         Q22 * dx * dy)
    
    return I


def track_point_lucas_kanade(frame1, frame2, point, window_size=15):
    """
    Track a single point from frame1 to frame2 using Lucas-Kanade.
    
    Returns:
        new_point: Tracked location in frame2
        flow_vector: (u, v) motion
    """
    x, y = point
    half = window_size // 2
    
    # Extract window
    x1, x2 = max(0, int(x) - half), min(frame1.shape[1], int(x) + half + 1)
    y1, y2 = max(0, int(y) - half), min(frame1.shape[0], int(y) + half + 1)
    
    I1 = frame1[y1:y2, x1:x2].astype(np.float32)
    I2 = frame2[y1:y2, x1:x2].astype(np.float32)
    
    # Compute gradients
    Ix = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3)
    It = I2 - I1
    
    # Flatten
    Ix, Iy, It = Ix.flatten(), Iy.flatten(), It.flatten()
    
    # Build A matrix and b vector
    A = np.vstack([Ix, Iy]).T
    b = -It
    
    # Solve least squares: A·[u,v] = b
    # d = (AᵀA)⁻¹Aᵀb
    AtA = A.T @ A
    if np.linalg.det(AtA) > 1e-5:  # Check invertibility
        Atb = A.T @ b
        flow = np.linalg.inv(AtA) @ Atb
        u, v = flow[0], flow[1]
    else:
        u, v = 0, 0
    
    new_x = x + u
    new_y = y + v
    
    return (new_x, new_y), (u, v)


def validate_tracking(video_path, num_points=5, frame_pair=(10, 11)):
    """
    Validate theoretical tracking against actual dense optical flow.
    """
    print("="*60)
    print("MOTION TRACKING VALIDATION")
    print("="*60)
    
    cap = cv2.VideoCapture(video_path)
    
    # Read frame pair
    for _ in range(frame_pair[0]):
        ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frames")
        return
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute dense flow for ground truth
    dense_flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Select random points with significant motion
    mag, _ = cv2.cartToPolar(dense_flow[..., 0], dense_flow[..., 1])
    motion_mask = mag > np.percentile(mag, 80)  # Top 20% motion regions
    
    y_coords, x_coords = np.where(motion_mask)
    indices = np.random.choice(len(y_coords), min(num_points, len(y_coords)), replace=False)
    
    points = [(x_coords[i], y_coords[i]) for i in indices]
    
    print(f"\nValidating {len(points)} points between frames {frame_pair[0]} and {frame_pair[1]}")
    print(f"{'Point':<10} {'Predicted (LK)':<25} {'Actual (Dense)':<25} {'Error':<10}")
    print("-"*70)
    
    errors = []
    results = []
    
    for i, (x, y) in enumerate(points):
        # Theoretical prediction (Lucas-Kanade)
        pred_point, flow_vec = track_point_lucas_kanade(gray1, gray2, (x, y))
        
        # Actual ground truth (dense flow at that point using bilinear interp)
        actual_u = bilinear_interpolate(dense_flow[..., 0], x, y)
        actual_v = bilinear_interpolate(dense_flow[..., 1], x, y)
        actual_point = (x + actual_u, y + actual_v)
        
        # Error
        error = np.sqrt((pred_point[0] - actual_point[0])**2 + 
                       (pred_point[1] - actual_point[1])**2)
        errors.append(error)
        
        print(f"({x:.0f},{y:.0f})    "
              f"({pred_point[0]:.2f}, {pred_point[1]:.2f})    "
              f"({actual_point[0]:.2f}, {actual_point[1]:.2f})    "
              f"{error:.3f}px")
        
        results.append({
            'initial': (x, y),
            'predicted': pred_point,
            'actual': actual_point,
            'error': error
        })
    
    print("-"*70)
    print(f"Mean tracking error: {np.mean(errors):.3f} pixels")
    print(f"Std tracking error: {np.std(errors):.3f} pixels")
    print(f"Max tracking error: {np.max(errors):.3f} pixels")
    
    # Visualization
    visualize_tracking(frame1, frame2, results)
    
    return results


def visualize_tracking(frame1, frame2, results):
    """Visualize tracking results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Frame 1 with initial points
    axes[0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    for r in results:
        x, y = r['initial']
        axes[0].plot(x, y, 'go', markersize=8)
    axes[0].set_title('Frame 1: Initial Points')
    axes[0].axis('off')
    
    # Frame 2 with predicted vs actual
    axes[1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    for r in results:
        # Predicted (red)
        px, py = r['predicted']
        axes[1].plot(px, py, 'ro', markersize=8, label='Predicted (LK)' if r == results[0] else "")
        
        # Actual (green)
        ax, ay = r['actual']
        axes[1].plot(ax, ay, 'g+', markersize=10, markeredgewidth=2, label='Actual (Dense)' if r == results[0] else "")
        
        # Connection line
        axes[1].plot([px, ax], [py, ay], 'y--', alpha=0.5)
    
    axes[1].set_title('Frame 2: Predicted vs Actual')
    axes[1].legend()
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('tracking_validation.png', dpi=150)
    print("\nSaved: tracking_validation.png")
    plt.show()


if __name__ == "__main__":
    # Print theoretical background
    theoretical_background()
    
    # Validate with two videos
    videos = ["data/video1.mov", "data/video2.mov"]
    
    for video in videos:
        if __import__('os').path.exists(video):
            print(f"\nProcessing: {video}")
            validate_tracking(video, num_points=5, frame_pair=(10, 11))
