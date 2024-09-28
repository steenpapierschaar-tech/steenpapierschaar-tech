import os
import glob
import cv2 as cv
import numpy as np

def maskBlueBG(img):
    """ Asssuming the background is blue, segment the image and return a
        BW image with foreground (white) and background (black)
    """ 
    # Change image color space
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Note that 0≤V≤1, 0≤S≤1, 0≤H≤360 and if H<0 then H←H+360
    # 8-bit images: V←255V,S←255S,H←H/2(to fit to 0 to 255)
    # see https://docs.opencv.org/4.5.3/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv

    # Define background color range in HSV space
    light_blue = (75,125,0)  # converted from HSV value obtained with colorpicker (150,50,0)
    dark_blue  = (140,255,255)  # converted from HSV value obtained with colorpicker (250,100,100)

    light_blue = (0,0,0)  # converted from HSV value obtained with colorpicker (150,50,0)
    dark_blue  = (255,50,255)  # converted from HSV value obtained with colorpicker (250,100,100)

    cv.imshow("img_hsv",img_hsv)
    cv.waitKey(0)

    # Mark pixels outside background color range
    mask = ~cv.inRange(img_hsv, light_blue, dark_blue)
    return mask


def maskWhiteBG(img):
    """ Asssuming the background is white, segment the image and return a
        BW image with foreground (white) and background (black)
    """ 
    cv.imshow("img",img)
    # Change image color space
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow("img_hsv",img_hsv)

    img_hsv = img_hsv[:,:,0]

    cv.imshow("img_hsv2",img_hsv)

    _, threshold = cv.threshold(img_hsv, 120,130 , cv.THRESH_BINARY_INV)
    _, threshold2 = cv.threshold(img_hsv, 70,120 , cv.THRESH_BINARY_INV)

    threshold3 = threshold2 + ~threshold

    _, threshold3 = cv.threshold(threshold3, 150, 255, cv.THRESH_BINARY)

    return threshold3

if __name__ == "__main__":
    """ Test segmentation functions"""
    data_path = r'G:\My Drive\data\gesture_data\hang_loose'
    data_path = r'G:\My Drive\data\gesture_data\thijs'

    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.png'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # loop over the image paths
    for filename in file_list:
        
        # load image and blur a bit
        img = cv.imread(filename)
        img = cv.blur(img,(3,3))        

        # mask background 
        mask = maskBlueBG(img)
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # show result and wait a bit        
        cv.imshow("Masked image", masked_img)
        k = cv.waitKey(1000) & 0xFF

        # if the `q` key or ESC was pressed, break from the loop
        if k == ord("q") or k == 27:
            break 
    
