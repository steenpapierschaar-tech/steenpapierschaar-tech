import os
import sys
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from dataset_loader import load_files
import argparse

def testGrabCut(image):

    img = cv.cvtColor(image,cv.COLOR_RGB2BGR)
    img[:,:,0] = 0
    img[:,:,1] = 0

    # Define an initial mask for GrabCut algorithm
    mask = np.zeros(img.shape[:2], np.uint8)

    # Create temporary arrays for GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle around the object (hand) for GrabCut
    rect = (1, 1, img.shape[1] - 2, img.shape[0] - 2)

    # Apply GrabCut
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 4, cv.GC_INIT_WITH_RECT)

    # Modify the mask so that sure foreground and possible foreground are set to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    image_no_bg = image * mask2[:, :, np.newaxis]

    image_no_bg = cv.cvtColor(image_no_bg,cv.COLOR_BGR2GRAY)

    #cv.imshow("result",image_no_bg)

    return image_no_bg



def prepimg(image):
    
    

   
    
    img = cv.cvtColor(image,cv.COLOR_RGB2BGR)
    img[:,:,0] = 0
    img[:,:,1] = 0
    #img[:,:,2] = 0
    #cv.boost
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(15,15))
    #gray  = clahe.apply(gray)

    gray = cv.GaussianBlur(gray,(9,9),0)

    gray = cv.convertScaleAbs(gray, alpha=1.9, beta=0)
    
    gray = extractpixels(gray) 
    gray = extractpixels(gray)
    gray = extractpixels(gray)
    gray = extractpixels(gray)
    gray = extractpixels(gray)

    gray = cv.convertScaleAbs(gray, alpha=2 , beta=0)
    gray = extractpixels(gray)
  
    gray = extractpixels(gray)
    gray = cv.convertScaleAbs(gray, alpha=2 , beta=0)
    gray = extractpixels(gray)
    gray = extractpixels(gray)
    gray = extractpixels(gray)
    gray = extractpixels(gray)
    gray = extractpixels(gray)

    gray = cv.GaussianBlur(gray,(5,5),0)
    kernel=  np.ones((6,6),np.uint8)
    canny = cv.Canny(gray,100,200)
    canny = cv.dilate(canny,kernel,1)
    
    
    gray[canny == 255]  = 0
    gray = cv.convertScaleAbs(gray, alpha=1.3 , beta=0)
    gray = cv.GaussianBlur(gray,(25,25),0)
   # gray = extractpixels(gray)
    
    cv.imshow("gray",gray)

    testjeroen(gray)
    
    gray = cv.GaussianBlur(gray,(7,7),0)
    
    cv.imshow("canny",canny)
    _, thresh = cv.threshold(gray,128  , 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU )
   # thresh = cv.dilate(thresh,kernel,1)
    cv.imshow("thres",thresh) 


    return  thresh

def extractpixels(gray):
    #gray = cv.GaussianBlur(gray,(3,3),0)
    noncappedpixels = gray [gray!= 255 & (gray != 0)]
    if len(noncappedpixels) > 0:
        min_val = np.min(noncappedpixels)
       
    
    
    output_image = np.copy(gray)

# Subtract the minimum value from all non-255 pixels
    output_image[output_image != 255 &  (output_image>10)] -= (min_val)

# Ensure pixel values don't go below 0
    output_image = np.clip(output_image, 0, 255)
    


    return output_image