"""
module3/convolution_theorem.py
Convolution Theorem Verification
Shows: Spatial Convolution ≡ Frequency Multiplication
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_gaussian_kernel(size, sigma):
    """Create 2D Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)  # Normalize to sum=1


def blur_spatial(image, kernel):
    """Method 1: Traditional convolution in spatial domain."""
    # Using filter2D with BORDER_WRAP to simulate periodic boundaries
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_WRAP)


def blur_fourier(image, kernel):
    """Method 2: Equivalent operation in frequency domain."""
    h, w = image.shape
    
    # Pad kernel to image size
    kernel_padded = np.zeros_like(image)
    kh, kw = kernel.shape
    
    # Place kernel at center
    start_y, start_x = (h - kh) // 2, (w - kw) // 2
    kernel_padded[start_y:start_y+kh, start_x:start_x+kw] = kernel
    
    # Shift kernel so center is at (0,0) for FFT
    kernel_padded = np.fft.ifftshift(kernel_padded)
    
    # FFT of image and kernel
    F = np.fft.fft2(image)
    H = np.fft.fft2(kernel_padded)
    
    # Multiplication in frequency domain
    G = F * H
    
    # Inverse FFT
    result = np.fft.ifft2(G)
    result = np.real(result)
    return np.clip(result, 0, 1) # Clip to valid range


def main():
    # Load grayscale image
    # because convolution theorem is typically demonstrated on single channel images
    img = cv2.imread('data/lemon.jpg', cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    
    # Create kernel
    kernel = create_gaussian_kernel(size=15, sigma=3)
    
    # Method 1: Spatial
    blur_s = blur_spatial(img, kernel)
    
    # Method 2: Frequency
    blur_f = blur_fourier(img, kernel)
    
    # PROOF OF EQUIVALENCE
    mse = np.mean((blur_s - blur_f)**2)
    max_diff = np.max(np.abs(blur_s - blur_f))
    
    print(f"Mean Squared Error: {mse:.2e}")
    print(f"Max Absolute Difference: {max_diff:.2e}")
    print("Theorem verified!" if mse < 1e-4 else "Discrepancy detected")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0,0].imshow(img, cmap='gray'); axes[0,0].set_title('Original')
    axes[0,1].imshow(kernel, cmap='viridis'); axes[0,1].set_title('Gaussian Kernel')
    axes[1,0].imshow(blur_s, cmap='gray'); axes[1,0].set_title('Spatial Convolution')
    axes[1,1].imshow(blur_f, cmap='gray'); axes[1,1].set_title('Frequency Multiplication')
    plt.tight_layout()
    plt.savefig('conv_theorem_verification.png')


if __name__ == "__main__":
    main()
