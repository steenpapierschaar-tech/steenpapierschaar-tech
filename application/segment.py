import os
import sys
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from datasetLoader import loadFiles
import argparse

def prepareImage(image):

    original = image.copy()
    
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image[:, :, 0] = 0  # Set the blue channel to 0
    image[:, :, 1] = 0  # Set the green channel to 0

    # Define an initial mask for GrabCut algorithm
    mask = np.zeros(image.shape[:2], np.uint8)

    # Create temporary arrays for GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle around the object (hand) for GrabCut
    rect = (1, 1, image.shape[1] - 2, image.shape[0] - 2)

    # Apply GrabCut
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 4, cv.GC_INIT_WITH_RECT)

    # Modify the mask so that sure foreground and possible foreground are set to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Apply the mask to the image
    image_no_bg = image * mask2[:, :, np.newaxis]
    
    image_no_bg = cv.cvtColor(image_no_bg, cv.COLOR_BGR2GRAY)

    # Threshold the image to get a binary mask
    _, mask = cv.threshold(image_no_bg, 1, 255, cv.THRESH_BINARY)
    
    # Apply morphological opening to remove small black spots inside the object
    kernel = np.ones((5, 5), np.uint8)

    # open
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)

    # dilate
    #dilated_mask = cv.dilate(mask, kernel, iterations=1)
    # erode
    #opened_mask = cv.erode(dilated_mask, kernel, iterations=1)
    
    # Show the mask
    cv.imshow("Mask", opened_mask)
    
    return opened_mask

def prepareImage2(image):

    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image[:, :, 0] = 0
    image[:, :, 1] = 0
    # image[:,:,2] = 0
    # cv.boost
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(15,15))
    # gray  = clahe.apply(gray)

    gray = cv.GaussianBlur(gray, (9, 9), 0)

    gray = cv.convertScaleAbs(gray, alpha=1.9, beta=0)

    # Show grey image
    gray = extractPixels(gray)
    gray = extractPixels(gray)
    gray = extractPixels(gray)
    gray = extractPixels(gray)
    gray = extractPixels(gray)
    gray = cv.convertScaleAbs(gray, alpha=2, beta=0)
    gray = extractPixels(gray)
    gray = extractPixels(gray)
    gray = cv.convertScaleAbs(gray, alpha=2, beta=0)
    gray = extractPixels(gray)
    gray = extractPixels(gray)
    gray = extractPixels(gray)
    gray = extractPixels(gray)
    gray = extractPixels(gray)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((6, 6), np.uint8)
    canny = cv.Canny(gray, 100, 200)
    canny = cv.dilate(canny, kernel, 1)

    gray[canny == 255] = 0
    gray = cv.convertScaleAbs(gray, alpha=1.3, beta=0)
    gray = cv.GaussianBlur(gray, (25, 25), 0)
    # gray = extractPixels(gray)

    #cv.imshow("gray", gray)

    # Not working this? 
    # testjeroen(gray)

    gray = cv.GaussianBlur(gray, (7, 7), 0)

    #cv.imshow("canny", canny)
    _, thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # thresh = cv.dilate(thresh,kernel,1)
    #cv.imshow("thres", thresh)

    return thresh

def extractPixels(gray):
    # gray = cv.GaussianBlur(gray,(3,3),0)
    noncappedpixels = gray[gray != 255 & (gray != 0)]
    if len(noncappedpixels) > 0:
        min_val = np.min(noncappedpixels)

    outputImage = np.copy(gray)

    # Subtract the minimum value from all non-255 pixels
    outputImage[outputImage != 255 & (outputImage > 10)] -= min_val

    # Ensure pixel values don't go below 0
    outputImage = np.clip(outputImage, 0, 255)

    return outputImage

def sobel(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
 
    # Apply the Sobel operator in the X direction
    sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3)

    # Apply the Sobel operator in the Y direction
    sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=3)

    # Combine the two gradients
    sobel_combined = cv.magnitude(sobel_x, sobel_y)

    # Normalize and convert the result to 8-bit image
    sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    _, mask = cv.threshold(sobel_combined, 50, 255, cv.THRESH_BINARY)
     
    return mask

def maskWhiteBG(image):
 
    # Blur the image to suppress noise
    image = cv.blur(image, (6, 6))

    # Convert the image to HSL color space
    image_hsl = cv.cvtColor(image, cv.COLOR_BGR2HLS)

    # Split the HSL channels
    h, l, s = cv.split(image_hsl)

    # Normalize the Lightness channel (L)
    l_normalized = cv.normalize(l, None, 0, 255, cv.NORM_MINMAX)

    # Merge the H, normalized L, and S channels back
    image_hsl = cv.merge([h, l_normalized, s])
    
    # Convert the image to HSL color space
    image_hsl = cv.cvtColor(image_hsl, cv.COLOR_BGR2HLS)
    
    # Define the HSL range for the white background (tune using trackbars in main)
    h_min = 85
    h_max = 160
    s_min = 0
    s_max = 200
    l_min = 120
    l_max = 255
    
    # Define the HSL range for the white background (tune using trackbars)
    lower_white = np.array([h_min, l_min, s_min])
    upper_white = np.array([h_max, l_max, s_max])

    # Threshold the image to mask the white background
    mask = cv.inRange(image_hsl, lower_white, upper_white)
    
    # Invert the mask to get the non-white areas
    mask_inv = cv.bitwise_not(mask)

    # return the treshold
    return mask_inv

def segmentImage(image):
    """
    Complete proces for image segmentation    
    """
    image_original = image.copy()
    prepared = prepareImage(image)
    mask = extractPixels(prepared)
    result = cv.bitwise_and(image_original, image_original, mask=prepared)
    
    return result

if __name__ == "__main__":
    """
    Test functions in this file
    """
    
    # load all dataset image files
    fileList = loadFiles()
    
    for filename in fileList:
        
        print("[INFO] processing image: {}".format(filename))
        
        # load and process the image
        
        image = cv.imread(filename)       
        original = image.copy()
        grabCutted = testGrabCut(image)
        prepared = prepareImage(image)
        extracted = extractPixels(prepared)
        sobeled = sobel(image)
        masked = cv.bitwise_and(original, original, mask=prepared)
        
        # Convert images back to RGB for visualization
        
        original = cv.cvtColor(original, cv.COLOR_BGR2RGB)
        grabCutted = cv.cvtColor(grabCutted, cv.COLOR_BGR2RGB)
        prepared = cv.cvtColor(prepared, cv.COLOR_BGR2RGB)
        extracted = cv.cvtColor(extracted, cv.COLOR_BGR2RGB)
        sobeled = cv.cvtColor(sobeled, cv.COLOR_BGR2RGB)
        masked = cv.cvtColor(masked, cv.COLOR_BGR2RGB)
        
        # Label the images
        
        cv.putText(original, 'Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(grabCutted, 'GrabCut', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(prepared, 'Prepared', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(extracted, 'Extracted', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(sobeled, 'Sobel', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(masked, 'Masked', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        
        # Show the images together
        
        row_A = cv.vconcat([original, grabCutted, prepared])
        row_B = cv.vconcat([extracted, sobeled, masked])
        combined = cv.hconcat([row_A, row_B])
        cv.imshow('Segmentation Tests', combined)
        
        # Go to the next image with any key press. Break loop if ESC is pressed
        k = cv.waitKey(0)
        if k == 27:
            break
        cv.destroyAllWindows()