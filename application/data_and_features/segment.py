import os
import sys
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from dataset_loader import load_files

def nothing(x):
    pass

def maskWhiteBG(img):

    # Blur the image to suppress noise
    img = cv.blur(img, (6, 6))
    
    # Create a copy of the image
    img_hsl_pre = img.copy()

    # Convert the image to HSL color space
    img_hsl = cv.cvtColor(img_hsl_pre, cv.COLOR_BGR2HLS)

    # Split the HSL channels
    h, l, s = cv.split(img_hsl)

    # Normalize the Lightness channel (L)
    l_normalized = cv.normalize(l, None, 0, 255, cv.NORM_MINMAX)

    # Merge the H, normalized L, and S channels back
    img_hsl = cv.merge([h, l_normalized, s])
    
    # Convert the image to HSL color space
    img_hsl = cv.cvtColor(img_hsl, cv.COLOR_BGR2HLS)
    
    # Define the HSL range for the white background (tune using trackbars in main)
    h_min = 85
    h_max = 200
    s_min = 0
    s_max = 200
    l_min = 90
    l_max = 255
    
    # Define the HSL range for the white background (tune using trackbars)
    lower_white = np.array([h_min, l_min, s_min])
    upper_white = np.array([h_max, l_max, s_max])

    # Threshold the image to mask the white background
    mask = cv.inRange(img_hsl, lower_white, upper_white)
    
    # Invert the mask to get the non-white areas
    mask_inv = cv.bitwise_not(mask)
    
    # Apply the mask to the image
    result = cv.bitwise_and(img, img, mask=mask_inv)

    # return the treshold
    return mask_inv

if __name__ == "__main__":
    
    # load all dataset image files
    file_list = load_files()
    
    for filename in file_list:
        print("[INFO] processing image: {}".format(filename))
        
        # load image
        img = cv.imread(filename)
        
        # Blur the image to suppress noise
        img = cv.blur(img, (6, 6))
        
        # Create a copy of the image
        img_hsl_pre = img.copy()
        
        # Convert the image to HSL color space
        img_hsl = cv.cvtColor(img_hsl_pre, cv.COLOR_BGR2HLS)

        # Split the HSL channels
        h, l, s = cv.split(img_hsl)

        # Normalize the Lightness channel (L)
        l_normalized = cv.normalize(l, None, 0, 255, cv.NORM_MINMAX)

        # Merge the H, normalized L, and S channels back
        img_hsl = cv.merge([h, l_normalized, s])

        # Create a window
        cv.namedWindow("White Background Masking (HSL)")

        # Create trackbars for adjusting Hue, Saturation, and Lightness
        cv.createTrackbar('Hue Min', 'White Background Masking (HSL)', 0, 255, nothing)
        cv.createTrackbar('Hue Max', 'White Background Masking (HSL)', 255, 255, nothing)
        cv.createTrackbar('Saturation Min', 'White Background Masking (HSL)', 0, 255, nothing)
        cv.createTrackbar('Saturation Max', 'White Background Masking (HSL)', 255, 255, nothing)
        cv.createTrackbar('Lightness Min', 'White Background Masking (HSL)', 200, 255, nothing)
        cv.createTrackbar('Lightness Max', 'White Background Masking (HSL)', 255, 255, nothing)

        while True:
            # Get the current positions of the trackbars
            h_min = cv.getTrackbarPos('Hue Min', 'White Background Masking (HSL)')
            h_max = cv.getTrackbarPos('Hue Max', 'White Background Masking (HSL)')
            s_min = cv.getTrackbarPos('Saturation Min', 'White Background Masking (HSL)')
            s_max = cv.getTrackbarPos('Saturation Max', 'White Background Masking (HSL)')
            l_min = cv.getTrackbarPos('Lightness Min', 'White Background Masking (HSL)')
            l_max = cv.getTrackbarPos('Lightness Max', 'White Background Masking (HSL)')

            # Define the HSL range for the white background (tune using trackbars)
            lower_white = np.array([h_min, l_min, s_min])
            upper_white = np.array([h_max, l_max, s_max])

            # Threshold the image to mask the white background
            mask = cv.inRange(img_hsl, lower_white, upper_white)
            # Invert the mask to get the non-white areas
            mask_inv = cv.bitwise_not(mask)
            # Apply the mask to the image
            result = cv.bitwise_and(img, img, mask=mask_inv)

            # Ensure all images have the same number of channels
            mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            mask_inv_bgr = cv.cvtColor(mask_inv, cv.COLOR_GRAY2BGR)

            # Show the mask and the masked image combined
            combined_result = cv.hconcat([img, mask_bgr, mask_inv_bgr, result])
            cv.imshow('White Background Masking (HSL)', combined_result)
            
            # Press 'q' or 'ESC' to exit the loop
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break


        # Destroy the windows after exiting
        cv.destroyAllWindows()
        
        # print the values
        print("Hue Min: {}".format(h_min), "Hue Max: {}".format(h_max), "Saturation Min: {}".format(s_min), "Saturation Max: {}".format(s_max), "Lightness Min: {}".format(l_min), "Lightness Max: {}".format(l_max))