import cv2 as cv
import numpy as np
import time
from dataset_loader import load_files

# Function to handle trackbar changes
def update_threshold(val):
    pass

# Function to find the largest contour
def find_largest_contour(contours):
    largest_contour = max(contours, key=cv.contourArea)
    return largest_contour

# Load all dataset image files
file_list = load_files()

for filename in file_list:
    print("[INFO] Processing image: {}".format(filename))
    
    # Load the image
    img = cv.imread(filename)
    img_original = img.copy()
      
    # Show image with boosted contrast
    cv.namedWindow('Boosted Contrast')
    cv.imshow('Boosted Contrast', img)
    
    # Apply Gaussian blur to the image
    img = cv.GaussianBlur(img, (5, 5), 0)
    
    # Create Otsu's thresholded image
    gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    
    scale = 5
    delta = 1.5
    ddepth = cv.CV_16S
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    gray = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # Apply thresholding to segment the background (white/gray)
    ret, threshold_otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    threshold_A = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    threshold_A_Mean = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    threshold_A_Gaussian = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    cv.namedWindow('Gradient')
    cv.imshow('Gradient', gray)

    # Invert threshold
    threshold_otsu = cv.bitwise_not(threshold_otsu)
    threshold_A = cv.bitwise_not(threshold_A)
    threshold_A_Mean = cv.bitwise_not(threshold_A_Mean)
    threshold_A_Gaussian = cv.bitwise_not(threshold_A_Gaussian)
    
    # Apply morphological operations to clean up the thresholded image
    kernel = np.ones((3,3), np.uint8)
    #threshold_otsu = cv.morphologyEx(threshold_otsu, cv.MORPH_OPEN, kernel)
    #threshold_A = cv.morphologyEx(threshold_A, cv.MORPH_OPEN, kernel)
    #threshold_A_Mean = cv.morphologyEx(threshold_A_Mean, cv.MORPH_OPEN, kernel)
    #threshold_A_Gaussian = cv.morphologyEx(threshold_A_Gaussian, cv.MORPH_OPEN, kernel)
    
    # Apply Dilation to the thresholded image
    threshold_otsu = cv.dilate(threshold_otsu, kernel, iterations=1)
    threshold_A = cv.dilate(threshold_A, kernel, iterations=1)
    threshold_A_Mean = cv.dilate(threshold_A_Mean, kernel, iterations=1)
    threshold_A_Gaussian = cv.dilate(threshold_A_Gaussian, kernel, iterations=1)
    
    # Apply Erosion to the thresholded image
    threshold_otsu = cv.erode(threshold_otsu, kernel, iterations=1)
    threshold_A = cv.erode(threshold_A, kernel, iterations=1)
    threshold_A_Mean = cv.erode(threshold_A_Mean, kernel, iterations=1)
    threshold_A_Gaussian = cv.erode(threshold_A_Gaussian, kernel, iterations=1)

    # Find the largest contour in the thresholded image
    contour_otsu, _ = cv.findContours(threshold_otsu, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_threshold_A, _ = cv.findContours(threshold_A, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_threshold_A_Mean, _ = cv.findContours(threshold_A_Mean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_threshold_A_Gaussian, _ = cv.findContours(threshold_A_Gaussian, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contours
    largest_contour_otsu = find_largest_contour(contour_otsu)
    largest_contour_threshold_A = find_largest_contour(contour_threshold_A)
    largest_contour_threshold_A_Mean = find_largest_contour(contour_threshold_A_Mean)
    largest_contour_threshold_A_Gaussian = find_largest_contour(contour_threshold_A_Gaussian)
    
    # Draw the largest contour on the original image
    img_contour_otsu = cv.drawContours(img_original.copy(), [largest_contour_otsu], -1, (0, 255, 0), 3)
    img_contour_threshold_A = cv.drawContours(img_original.copy(), [largest_contour_threshold_A], -1, (0, 255, 0), 3)
    img_contour_threshold_A_Mean = cv.drawContours(img_original.copy(), [largest_contour_threshold_A_Mean], -1, (0, 255, 0), 3)
    img_contour_threshold_A_Gaussian = cv.drawContours(img_original.copy(), [largest_contour_threshold_A_Gaussian], -1, (0, 255, 0), 3)

    # Show the drawn contours
    cv.namedWindow('Contours')
    cv.imshow('Contours', cv.hconcat([img_contour_otsu, img_contour_threshold_A, img_contour_threshold_A_Mean, img_contour_threshold_A_Gaussian]))

    # Show the image
    cv.namedWindow('Hand Segmentation')
    cv.imshow('Hand Segmentation', cv.hconcat([threshold_otsu, threshold_A, threshold_A_Mean, threshold_A_Gaussian]))
    
    cv.waitKey(0)

    # Exit the loop on pressing 'q' or 'ESC'
    key = cv.waitKey(0) & 0xFF
    if key == ord('q') or key == 27:
        break

# Destroy all windows after exiting
cv.destroyAllWindows()
