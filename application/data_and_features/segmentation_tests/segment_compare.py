import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from application.datasetLoader import load_files

# This might be interesting to use as a reference
# https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

normalization_method = 1
smoothing_method = 2
threshold_method = 4
morph_method = 4
edge_detection_method = 1
background_removal_method = 1

def update_normalization_method(val):
    global normalization_method
    normalization_method = val
    
def update_smoothing_method(val):
    global smoothing_method
    smoothing_method = val

def update_threshold_method(val):
    global threshold_method
    threshold_method = val
    
def update_morphological_method(val):
    global morph_method
    morph_method = val
    
def update_edge_detection_method(val):
    global edge_detection_method
    edge_detection_method = val

def update_background_removal_method(val):
    global background_removal_method
    background_removal_method = val

def normalize_image(image):
    """
    Normalize the image using CLAHE, histogram equalization and normalization
    """
    
    # Named window for all the images
    cv.namedWindow('1 Normalization')
    
    # Create trackbars to choose which image to return. Values range from 1 to 4
    cv.createTrackbar('Method', '1 Normalization', normalization_method, 4, update_normalization_method)
    
    # Apply CLAHE to the image
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_h = clahe.apply(h)
    clahe_s = clahe.apply(s)
    clahe_v = clahe.apply(v)
    clahe = cv.merge([clahe_h, clahe_s, clahe_v])
    clahe = cv.cvtColor(clahe, cv.COLOR_HSV2BGR)
    
    # Apply histogram equalization to 3 hsv channels
    equalized_h = cv.equalizeHist(h)
    equalized_s = cv.equalizeHist(s)
    equalized_v = cv.equalizeHist(v)
    equalized = cv.merge([equalized_h, s, equalized_v])
    equalized = cv.cvtColor(equalized, cv.COLOR_HSV2BGR)
    
    # Apply normalization to 3 hsv channels
    normalized_h = cv.normalize(h, None, 0, 255, cv.NORM_MINMAX)
    normalized_s = cv.normalize(s, None, 0, 255, cv.NORM_MINMAX)
    normalized_v = cv.normalize(v, None, 0, 255, cv.NORM_MINMAX)
    normalized = cv.merge([normalized_h, normalized_s, normalized_v])
    normalized = cv.cvtColor(normalized, cv.COLOR_HSV2BGR)
    
    # Label the images
    image_label = image.copy()
    clahe_label = clahe.copy()
    equalized_label = equalized.copy()
    normalized_label = normalized.copy()
    
    cv.putText(image_label, '1 Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(clahe_label, '2 CLAHE', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(equalized_label, '3 Equalized', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(normalized_label, '4 Normalized', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    
    # Show the images together
    row_A = cv.vconcat([image_label, clahe_label, equalized_label, normalized_label])

    # Display the image 
    cv.imshow('1 Normalization', row_A)
    
    # Get the trackbar position for returning the image
    selection = cv.getTrackbarPos('Method', '1 Normalization')
    match selection:
        case 1:
            normalized_image = image
        case 2:
            normalized_image = clahe
        case 3:
            normalized_image = equalized
        case 4:
            normalized_image = normalized
        case _:
            normalized_image = image

    return normalized_image

def smooth_image(image):
    # Smooth the image
    # Smooth image, bilateral filter for sharp edges or gaussian blur

    # Create window for smoothing
    cv.namedWindow('2 Smoothing')
    
    # Create trackbars for choosing the smoothing method
    cv.createTrackbar('Method', '2 Smoothing', smoothing_method, 3, update_smoothing_method)
    
    bilateral = cv.bilateralFilter(image, 9, 75, 75)
    gaussian = cv.GaussianBlur(image, (9, 9), 0)
    
    # Label the images
    image_label = image.copy()
    bilateral_label = bilateral.copy()
    gaussian_label = gaussian.copy()
    
    cv.putText(image_label, '1 Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(bilateral_label, '2 Bilateral', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(gaussian_label, '3 Gaussian', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    
    # Show the images together
    row_A = cv.vconcat([image_label, bilateral_label, gaussian_label])
    cv.imshow('2 Smoothing', row_A)
    
    # Get the trackbar position for returning the image
    selection = cv.getTrackbarPos('Method', '2 Smoothing')
    match selection:
        case 2:
            smoothed = bilateral
        case 3:
            smoothed = gaussian
        case _:
            smoothed = image

    return smoothed

def threshold_image(image):
    # Threshold the image
    # Binary thresholding, adaptive mean thresholding, adaptive Gaussian thresholding, otsu thresholding
    
    # Convert to grayscale for thresholding
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Create window for thresholding including trackbars
    cv.namedWindow('3 Thresholding')
    cv.createTrackbar('Method', '3 Thresholding', threshold_method, 5, update_threshold_method)
    
    binary_threshold = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    adaptive_mean_threshold = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 2)
    adaptive_gaussian_threshold = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 2)
    otsu_threshold, otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Label the images
    image_label = image.copy()
    binary_label = binary_threshold[1].copy()
    adaptive_mean_label = adaptive_mean_threshold.copy()
    adaptive_gaussian_label = adaptive_gaussian_threshold.copy()
    otsu_label = otsu.copy()
    
    cv.putText(image_label, '1 Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(binary_label, '2 Binary', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(adaptive_mean_label, '3 Adaptive Mean', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(adaptive_gaussian_label, '4 Adaptive Gaussian', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(otsu_label, '5 Otsu', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    
    # Show the images together
    row_A = cv.vconcat([image_label, binary_label, adaptive_mean_label, adaptive_gaussian_label, otsu_label])
    cv.imshow('3 Thresholding', row_A)
    
    # Get the trackbar position for returning the image
    selection = cv.getTrackbarPos('Method', '3 Thresholding')
    match selection:
        case 2:
            thresholded = binary_threshold[1]
        case 3:
            thresholded = adaptive_mean_threshold
        case 4:
            thresholded = adaptive_gaussian_threshold
        case 5:
            thresholded = otsu
        case _:
            thresholded = image

    return thresholded

def morphological_operations(image):
    # Apply morphological operations to the image
    # Erosion, dilation, opening, closing, gradient, tophat, blackhat
    
    # Create window for morphological operations
    cv.namedWindow('4 Morphological Operations')
    
    # Create trackbars for choosing the morphological operation
    cv.createTrackbar('Method', '4 Morphological Operations', morph_method, 6, update_morphological_method)
    
    kernel = np.ones((5, 5), np.uint8)
    
    erosion = cv.erode(image, kernel, iterations=1)
    dilation = cv.dilate(image, kernel, iterations=1)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    
    # Erode and then dilate
    custom_morph = cv.erode(image, kernel, iterations=2)
    custom_morph = cv.dilate(custom_morph, kernel, iterations=2)
    
    # Label the images
    image_label = image.copy()
    erosion_label = erosion.copy()
    dilation_label = dilation.copy()
    opening_label = opening.copy()
    closing_label = closing.copy()
    custom_morph_label = custom_morph.copy()
    
    cv.putText(image_label, '1 Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(erosion_label, '2 Erosion', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(dilation_label, '3 Dilation', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(opening_label, '4 Opening', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(closing_label, '5 Closing', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(custom_morph_label, '6 Custom Morph', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

    # Show the images together
    row_A = cv.vconcat([image_label, erosion_label, dilation_label, opening_label, closing_label, custom_morph_label])
    cv.imshow('4 Morphological Operations', row_A)
    
    # Get the trackbar position for returning the image
    selection = cv.getTrackbarPos('Method', '4 Morphological Operations')
    match selection:
        case 2:
            morphed = erosion
        case 3:
            morphed = dilation
        case 4:
            morphed = opening
        case 5:
            morphed = closing
        case 6:
            morphed = custom_morph
        case _:
            morphed = image
            
    return morphed

def edge_detection(image):
    # Detect edges in the image
    # Edge detection, which type? Canny, Sobel, Laplacian
    
    # Create window for edge detection
    cv.namedWindow('5 Edge Detection')
    cv.createTrackbar('Method', '5 Edge Detection', edge_detection_method, 4, update_edge_detection_method)
    
    canny = cv.Canny(image, 100, 200)
    sobel = cv.Sobel(image, cv.CV_64F, 1, 1, ksize=5)
    laplacian = cv.Laplacian(image, cv.CV_64F)
    
    # Convert Sobel and Laplacian to 8-bit
    sobel = cv.convertScaleAbs(sobel)
    laplacian = cv.convertScaleAbs(laplacian)
    
    # Label the images
    image_label = image.copy()
    canny_label = canny.copy()
    sobel_label = sobel.copy()
    laplacian_label = laplacian.copy()
    
    cv.putText(image_label, '1 Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(canny_label, '2 Canny', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(sobel_label, '3 Sobel', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(laplacian_label, '4 Laplacian', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    
    # Show the images together
    row_A = cv.vconcat([image_label, canny_label, sobel_label, laplacian_label])
    cv.imshow('5 Edge Detection', row_A)
    
    # Get the trackbar position for returning the image
    selection = cv.getTrackbarPos('Method', '5 Edge Detection')
    match selection:
        case 2:
            edges = canny
        case 3:
            edges = sobel
        case 4:
            edges = laplacian
        case _:
            edges = image

    return edges

def background_removal(original_image, image):
    # Remove the background from the image
    
    # Find the second largest contour in the image
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    
    # Create a mask for the largest contour
    mask = np.zeros_like(image)
    cv.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # Invert the mask
    mask = cv.bitwise_not(mask)
    
    # Apply the mask to the original image
    background_removed = cv.bitwise_and(original_image, original_image, mask=mask)
    
    # Draw the largest contour on the original image
    cv.drawContours(original_image, [largest_contour], -1, (0, 255, 0), 2)
    
    # Display the original and segmented image
    original_display = original_image.copy()
    segmented_display = background_removed.copy()
    
    cv.putText(original_display, 'Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(segmented_display, 'Segmented', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    
    # Show the images together
    row_A = cv.vconcat([original_display, segmented_display])
    cv.imshow('6 Background Removal', row_A)
    
    return background_removed

def segmentate(image):
    """
    Segmentate the image
    """

    normalized = normalize_image(image)
    
    smoothed = smooth_image(normalized)
    
    thresholded = threshold_image(smoothed)
    
    morphed = morphological_operations(thresholded)
    
    edges = edge_detection(morphed)
        
    background_removed = background_removal(image, edges)
    
    return background_removed

if __name__ == "__main__":

    # Building the file list
    file_list = load_files()
    
    # sort files by filename
    file_list.sort()
    
    
    for filename in file_list:
        # Print the filename
        print("[INFO] processing image: {}".format(filename))
        
        # load image
        image = cv.imread(filename)
        
        # process the image
        segmented = segmentate(image)
        
        # Go to the next image with any key press. Break loop if ESC is pressed
        k = cv.waitKey(0)
        if k == 27:
            break
        
    cv.destroyAllWindows()
    