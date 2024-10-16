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
    img[:,:,1] = img[:,:,1] *0.1
    #img[:,:,2] = 0
    #cv.boost
    
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    blurred = cv.GaussianBlur(gray,(25,25),0)
    gray = cv.equalizeHist(gray)
    cv.imshow("gray",gray)
    
    
    #canny = cv.Canny(blurred,100,200)
    _, thresh = cv.threshold(blurred,200, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    


    return  thresh