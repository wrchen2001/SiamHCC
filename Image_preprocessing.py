"""
Chinese Character Quality Assessment - Image Preprocessing Step
Processing steps: Bilateral filtering → Manual weighted grayscale → Global threshold binarization
Dependencies: OpenCV 4.x, NumPy
"""

import cv2
import numpy as np
import os

def preprocess_character_image(input_path, output_path, 
                              bilateral_d=40, bilateral_sigma_color=75, bilateral_sigma_space=75,
                              threshold=150):
    """
    Standardized preprocessing pipeline for Chinese character images
    
    Parameters:
        input_path: Path to input image (str)
        output_path: Path to save processed image (str)
        bilateral_d: Bilateral filter diameter (int, default=40)
        bilateral_sigma_color: Color space sigma (int, default=75)
        bilateral_sigma_space: Spatial space sigma (int, default=75)
        threshold: Binarization threshold (0-255, default=150)
        
    Returns:
        None (Saves processed image directly to output path)
        
    Raises:
        ValueError: Invalid input path or image read failure
        RuntimeError: Image processing error
    """
    # Validate input path
    if not os.path.exists(input_path):
        raise ValueError(f"Input path missing: {input_path}")
    
    # Read image with OpenCV
    original_img = cv2.imread(input_path)
    if original_img is None:
        raise ValueError(f"Image read failed: {input_path}")

    try:
        # Step 1: Bilateral Filtering (original parameters)
        filtered = cv2.bilateralFilter(
            original_img,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space
        )

        # Step 2: Averaged-weighted Grayscale Conversion
        height, width = filtered.shape[:2]
        gray_3channel = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Pixel-wise weighted calculation (BGR format)
        for i in range(height):
            for j in range(width):
                B, G, R = filtered[i, j]
                gray_val = 0.114*B + 0.587*G + 0.299*R  # Standard coefficients
                gray_3channel[i, j] = [gray_val]*3

        # Step 3: Binarization with Threshold
        # Convert to single-channel using max value (redundant but preserves original logic)
        gray_single = cv2.cvtColor(gray_3channel, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_single, threshold, 255, cv2.THRESH_BINARY)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save final binary image
        cv2.imwrite(output_path, binary)
        
    except Exception as e:
        raise RuntimeError(f"Processing error: {str(e)}")