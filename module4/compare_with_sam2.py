"""
module4/compare_with_sam2.py
SAM2 Comparison - Load pre-generated SAM2 masks
"""

import cv2
import numpy as np


def load_sam2_mask(path):
    """Load SAM2 mask (assumes binary image saved from SAM2)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"SAM2 mask not found: {path}")
    # Binarize (SAM2 outputs may be 0-255 or already 0-1)
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_binary


def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union.
    Masks should be binary (0 or 255).
    """
    # Ensure same size
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
    
    # Convert to boolean
    m1 = mask1 > 0
    m2 = mask2 > 0
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    iou = intersection / union if union > 0 else 0
    return iou, intersection, union


def visualize_comparison(image, opencv_mask, sam2_mask, contour=None):
    """Side-by-side comparison of methods."""
    import matplotlib.pyplot as plt

    # Ensure all masks match image size
    h, w = image.shape[:2]
    if opencv_mask.shape != (h, w):
        opencv_mask = cv2.resize(opencv_mask, (w, h))
    if sam2_mask.shape != (h, w):
        sam2_mask = cv2.resize(sam2_mask, (w, h))
    
    # Create overlay images
    opencv_overlay = cv2.addWeighted(image, 0.5, 
                                   cv2.cvtColor(opencv_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
    sam2_overlay = cv2.addWeighted(image, 0.5,
                                   cv2.cvtColor(sam2_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
    
    # Difference map (XOR)
    diff = cv2.bitwise_xor(opencv_mask, sam2_mask)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('Original')
    
    axes[0,1].imshow(opencv_mask, cmap='gray')
    axes[0,1].set_title('OpenCV Classical Method')
    
    axes[0,2].imshow(sam2_mask, cmap='gray')
    axes[0,2].set_title('SAM2 (Deep Learning)')
    
    axes[1,0].imshow(cv2.cvtColor(opencv_overlay, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title('OpenCV Method Overlay')
    
    axes[1,1].imshow(cv2.cvtColor(sam2_overlay, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title('SAM2 Overlay')
    
    axes[1,2].imshow(diff, cmap='hot')
    axes[1,2].set_title('Difference (XOR)')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    # Paths
    image_path = "data/giraffe thermal.jpg"
    opencv_mask_path = "output_giraffe_mask.png"  # From classical OpenCV method
    sam2_mask_path = "output_giraffe_SAM2.png"     # From SAM2 (manual save)
    
    # Load
    image = cv2.imread(image_path)
    opencv_mask = cv2.imread(opencv_mask_path, cv2.IMREAD_GRAYSCALE)
    sam2_mask = load_sam2_mask(sam2_mask_path)
    
    # Compare
    iou, inter, union = compute_iou(opencv_mask, sam2_mask)
    
    print(f"{'='*50}")
    print("SAM2 vs Classical Comparison")
    print(f"{'='*50}")
    print(f"Intersection: {inter} pixels")
    print(f"Union:        {union} pixels")
    print(f"IoU Score:    {iou:.4f} ({iou*100:.2f}%)")
    print(f"{'='*50}")
    
    if iou > 0.7:
        print("✓ Excellent agreement")
    elif iou > 0.5:
        print("~ Moderate agreement")
    else:
        print("✗ Poor agreement - methods diverge significantly")
    
    # Visualize
    visualize_comparison(image, opencv_mask, sam2_mask)
    
    return iou


if __name__ == "__main__":
    main()
