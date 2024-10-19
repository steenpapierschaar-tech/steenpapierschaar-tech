import os
import sys
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from dataset_loader import load_files
import argparse

def prepimg(image):
    img = cv.cvtColor(image,cv.COLOR_RGB2BGR)
    img[:,:,0] = 0
    img[:,:,1] = 0
    #img[:,:,2] = 0
    #cv.boost
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(15,15))
    #gray  = clahe.apply(gray)

    gray = cv.GaussianBlur(gray,(5,5),0)

    gray = cv.convertScaleAbs(gray, alpha=5.3 , beta=0)
   
    gray = extractpixels(gray) 
    gray = cv.convertScaleAbs(gray, alpha=1.1 , beta=0)
    gray = extractpixels(gray)
    gray = cv.convertScaleAbs(gray, alpha=1.1 , beta=0)
    gray = extractpixels(gray)
    kernel=  np.ones((3,3),np.uint8)
    gray = cv.dilate(gray,kernel,1)
    
    cv.imshow("gray",gray)
    
    
    #canny = cv.Canny(blurred,100,200)
    _, thresh = cv.threshold(gray,175  , 255, cv.THRESH_BINARY_INV )
    
    cv.imshow("thres",thresh) 


    return  thresh

def extractpixels(gray):
    #gray = cv.GaussianBlur(gray,(3,3),0)
    noncappedpixels = gray [gray!= 255]
    if len(noncappedpixels) > 0:
        min_val = np.min(noncappedpixels)
    
    
    output_image = np.copy(gray)

# Subtract the minimum value from all non-255 pixels
    output_image[output_image != 255] -= (min_val)

# Ensure pixel values don't go below 0
    output_image = np.clip(output_image, 0, 255)
    


    return output_image