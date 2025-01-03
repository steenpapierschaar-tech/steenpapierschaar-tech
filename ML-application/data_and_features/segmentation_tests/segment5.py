import cv2 as cv
import numpy as np
import time
from fileHandler import load_files

# Function to handle trackbar changes
def update_threshold(val):
    pass

# Function to find the largest contour
def find_largest_contour(contours):
    largest_contour = max(contours, key=cv.contourArea)
    return largest_contour

# Load all dataset image files
file_list = load_files()

cv.namedWindow('Image processing with Sobel and Otsu')
cv.createTrackbar('Scale', 'Image processing with Sobel and Otsu', 1, 10, update_threshold)
cv.createTrackbar('Delta', 'Image processing with Sobel and Otsu', 1, 10, update_threshold)
cv.createTrackbar('Kernel Size', 'Image processing with Sobel and Otsu', 1, 10, update_threshold)

for filename in file_list:
    print("[INFO] Processing image: {}".format(filename))
    

        
    
    # Load the image
    img = cv.imread(filename)
    img_original = img.copy()
      
    # Apply Gaussian blur to the image
    img = cv.GaussianBlur(img, (5, 5), 0)
    
    # Create Otsu's thresholded image
    gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    
    while(True):
        scale = cv.getTrackbarPos('Scale', 'Image processing with Sobel and Otsu')
        delta = cv.getTrackbarPos('Delta', 'Image processing with Sobel and Otsu')
        ksize = cv.getTrackbarPos('Kernel Size', 'Image processing with Sobel and Otsu')
        
        if ksize % 2 == 0:
            ksize += 1
        
        if ksize < 3:
            ksize = 3
            
        print("[INFO] Scale: {}, Delta: {}, Kernel Size: {}".format(scale, delta, ksize))
    
        # Create Sobel gradient image
        ddepth = cv.CV_16S
        grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
        # Apply thresholding to segment the background (white/gray)
        ret, threshold_otsu = cv.threshold(sobel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # Invert threshold
        threshold_otsu = cv.bitwise_not(threshold_otsu)
        
        # Find the largest contour in the thresholded image
        contour_otsu, _ = cv.findContours(threshold_otsu, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Get the largest contours
        largest_contour_otsu = find_largest_contour(contour_otsu)

        # Draw the largest contour on the original image
        img_contour_otsu = cv.drawContours(img_original.copy(), [largest_contour_otsu], -1, (0, 255, 0), 3)
        
        # Convert otsu threshold to RGB
        sobel_BGR = cv.cvtColor(sobel, cv.COLOR_GRAY2BGR)
        otsu_threshold_BGR = cv.cvtColor(threshold_otsu, cv.COLOR_GRAY2BGR)
        
        # Label the images
        cv.putText(img_original, 'Original Image', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.putText(sobel_BGR, 'Sobel Gradient', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.putText(otsu_threshold_BGR, 'Otsu Threshold', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.putText(img_contour_otsu, 'Largest Contour', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Show the processed images
        cv.namedWindow('Image processing with Sobel and Otsu')
        row_A = cv.hconcat([img_original, sobel_BGR, otsu_threshold_BGR, img_contour_otsu])
        cv.imshow('Image processing with Sobel and Otsu', row_A)

        cv.waitKey(0)

        # Exit the loop on pressing 'q' or 'ESC'
        key = cv.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break

# Destroy all windows after exiting
cv.destroyAllWindows()