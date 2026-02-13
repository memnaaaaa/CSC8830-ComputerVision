"""
Thermal Animal Segmentation - Classical Computer Vision
Finds exact animal boundaries using OpenCV (no ML/DL)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def segment_thermal_animal(image_path, visualize=True):
    """
    Segment animal from thermal image using classical CV techniques.
    
    Args:
        image_path: Path to thermal image
        visualize: Show intermediate steps
    
    Returns:
        contour: Largest contour representing animal boundary
        mask: Binary segmentation mask
        processed: Dictionary of intermediate images
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert to grayscale (thermal is often pseudocolored, so we need luminance)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    processed = {'original': img, 'gray': gray}
    
    # Step 1: Noise reduction
    # Gaussian blur to smooth sensor noise while preserving edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    processed['blurred'] = blurred
    
    # Step 2: Segmentation using Otsu's thresholding
    mask = segment_bimodal(blurred, processed)
    processed['initial_mask'] = mask
    
    # Step 3: Morphological cleanup
    mask_clean = morphological_refinement(mask)
    processed['clean_mask'] = mask_clean
    
    # Step 4: Contour extraction
    contour = extract_largest_contour(mask_clean)
    
    # Step 5: Boundary refinement (optional - for "exact" boundaries)
    if contour is not None:
        refined_contour = refine_boundary(gray, contour)
    else:
        refined_contour = None
    
    if visualize:
        visualize_results(processed, refined_contour)
    
    return refined_contour, mask_clean, processed


def segment_bimodal(gray, processed):
    """
    Otsu's thresholding for bimodal thermal images.
    Assumes animal is brighter than background.
    """
    thresh, binary = cv2.threshold(gray, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu threshold: {thresh}")
    return binary


def morphological_refinement(mask):
    """Clean up segmentation mask."""
    kernel_small = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    kernel_medium = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    cleaned = cv2.medianBlur(closed, 5)
    
    return cleaned


def extract_largest_contour(mask):
    """Find the largest contour (the animal)."""
    contours, hierarchy = cv2.findContours(mask, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found!")
        return None
    
    largest = max(contours, key=cv2.contourArea)
    print(f"Largest contour area: {cv2.contourArea(largest)} pixels")
    
    return largest


def refine_boundary(gray, contour):
    """
    Refine contour for exact boundaries.
    Uses active contours (snakes) or edge snapping.
    """
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    edges = cv2.Canny(gray, 50, 150)
    
    edge_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    contour_band = cv2.dilate(mask, np.ones((15,15), np.uint8), iterations=1)
    contour_band = cv2.bitwise_and(contour_band, cv2.bitwise_not(mask))
    
    snapped_edges = cv2.bitwise_and(edges, contour_band)
    
    edge_contours, _ = cv2.findContours(snapped_edges, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    if edge_contours:
        best_edge = max(edge_contours, key=cv2.contourArea)
        combined = np.vstack([contour, best_edge])
        refined = cv2.convexHull(combined)
        return refined
    
    return contour


def visualize_results(processed, final_contour):
    """Display pipeline stages."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for name, img in processed.items():
        if plot_idx >= 6:
            break
            
        if len(img.shape) == 2:
            axes[plot_idx].imshow(img, cmap='gray')
        else:
            axes[plot_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        axes[plot_idx].set_title(name)
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Final result with contour
    if final_contour is not None and 'original' in processed:
        result = processed['original'].copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result, [final_contour], -1, (0, 255, 0), 2)
        axes[plot_idx].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[plot_idx].set_title('Final Boundary')
        axes[plot_idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    image = [
        ('data/giraffe thermal.jpg', 'giraffe'),
    ]
    
    for path, animal_type in image:
        print(f"\n{'='*50}")
        print(f"Processing: {path}")
        print(f"{'='*50}")
        
        try:
            contour, mask, stages = segment_thermal_animal(
                path,
                visualize=True
            )
            
            if contour is not None:
                # Save results
                cv2.imwrite(f'output_{animal_type}_mask.png', mask)
                print(f"Saved: output_{animal_type}_mask.png")
            else:
                print("Segmentation failed - no contour found")
                
        except Exception as e:
            print(f"Error processing {path}: {e}")


if __name__ == "__main__":
    main()
