import os
import sys
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from application.datasetLoader import load_files

if __name__ == "__main__":
    
    # load all dataset image files
    file_list = load_files()
    
    for filename in file_list:
        print("[INFO] processing image: {}".format(filename))
        
        # load image
        img = cv.imread(filename)
        
        # Blur the image to suppress noise
        img = cv.blur(img, (6, 6))
                
        image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB for visualization
        
        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply thresholding to segment the background (white/gray)
        _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Invert the thresholded image (hand becomes white, background becomes black)
        binary_inv = cv.bitwise_not(binary)

        # Create a mask with binary_inv as an initial mask
        mask = np.where(binary_inv > 0, 1, 0).astype('uint8')  # Foreground is 1, background is 0

        # Create temporary arrays used by the GrabCut algorithm
        bgdModel = np.zeros((1, 65), np.float64)  # Background model
        fgdModel = np.zeros((1, 65), np.float64)  # Foreground model

        # Use the binary_inv as an initial mask for GrabCut
        cv.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

        # Modify the mask such that background is 0 and foreground is 1
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Mask the image to only show the foreground (the hand)
        hand_segmented = image_rgb * mask2[:, :, np.newaxis]
        
        # Optional: Perform morphological operations to clean up the mask
        kernel = np.ones((9,9), np.uint8)
        mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)
        
        # Mask the image again with the refined mask
        hand_segmented = image_rgb * mask2[:, :, np.newaxis]

        # Display the original and segmented image
        plt.figure(figsize=(10, 5))
        plt.subplot(221), plt.imshow(image_rgb), plt.title('Original Image')
        plt.subplot(222), plt.imshow(hand_segmented), plt.title('Hand Segmented')
        plt.subplot(223), plt.imshow(mask), plt.title('Initial Mask')
        plt.subplot(224), plt.imshow(mask2), plt.title('Refined Mask')
        # make plot tight
        plt.tight_layout()
        plt.show()
       
        # Press 'q' or 'ESC' to exit the loop
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        # Destroy the windows after exiting
        cv.destroyAllWindows()
