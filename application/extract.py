import os
import glob
import cv2 as cv
import numpy as np
from segment import *
from datasetLoader import loadFiles

def getLargestContour(img_BW):
    """ Return largest contour in foreground as an nd.array """
    contours, hier = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL,
                                     cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)
    
    return np.squeeze(contour)

def getContourExtremes(contour):
    """ Return contour extremes as an tuple of 4 tuples """
    # determine the most extreme points along the contour
    left = contour[contour[:, 0].argmin()]
    right = contour[contour[:, 0].argmax()]
    top = contour[contour[:, 1].argmin()]
    bottom = contour[contour[:, 1].argmax()]

    return np.array((left, right, top, bottom))

def getConvexityDefects(contour):
    """ Return convexity defects in a contour as an nd.array """
    hull = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull)
    if defects is not None:
        defects = defects.squeeze()

    return defects

def getFeatures(contour):
    """ Return some simple contour features
        See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
    """ 
    
    # get bounding box
    x,y,w,h = cv.boundingRect(contour)

    # basic contour features
    area = cv.contourArea(contour)

    # get contour length
    contourLength = cv.arcLength(contour, True)

    # get convex hull
    ConvexHull  = cv.convexHull(contour, returnPoints=True)
    
    # Get convex hull length
    ConvexHullLength = cv.arcLength(ConvexHull, True)
    
    # get the top 2 convexity defects
    defects = getConvexityDefects(contour)
    
    if defects is None or defects.size == 0:
        print("No convexity defects found!")
        ConvexityDefects = 0.0
    
    if defects is not None and defects.size > 0:
        # Select only the 2 largest defects
        ConvexityDefects = np.flip(np.sort(defects))[:2]
        # Get average of top 2 defects
        ConvexityDefects = np.mean(ConvexityDefects)
    
    compactness = contourLength/area
    
    moments = cv.moments(contour)
    huMoment = cv.HuMoments(moments)
    huMoment = -1.0 * np.sign(huMoment) * np.log10(abs(huMoment))
    
    circularity = (4*np.pi*area)/(contourLength**2)
    
    aspectRatio = float(w)/h
    
    extent = float(area)/(w*h)

    # compile a feature vector
    features = np.array([area, contourLength, ConvexHullLength, ConvexityDefects, compactness, circularity, aspectRatio, extent])

    return (features)

def getSimpleContourFeatures(contour):
    """ Return some simple contour features
        See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
    """       
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    x,y,w,h = cv.boundingRect(contour)
    aspect_ratio = float(w)/h
    rect_area = w*h
    extent = float(area)/rect_area
    features = np.array((area, perimeter, aspect_ratio, extent))
    
    return (features)

def getContourFeatures(contour):
    """ Return some contour features
    """    
    # basic contour features
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    extremePoints = getContourExtremes(contour)

    # get contour convexity defect depths      
    # see https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    defects = getConvexityDefects(contour)

    if defects is not None and defects.ndim > 1:
        defect_depths = defects[:, -1] / 256.0
    else:
        defect_depths = np.zeros(6)

    # select only the 6 largest depths
    defect_depths = np.flip(np.sort(defect_depths))[0:6]

    # compile a feature vector
    features = np.append(defect_depths, (area, perimeter))

    return (features, defects)

def getHuMoments(moments):
    """ Return scaled Hu Moments """
    huMoments = cv.HuMoments(moments)
    scaled_huMoments = -1.0 * np.sign(huMoments) * np.log10(abs(huMoments))

    return np.squeeze(scaled_huMoments)

def getBlobFeatures(img_BW):
    """ Asssuming a BW image with a single white blob on a black background,
        return some blob features.
    """    
    # scaled invariant moments
    moments = cv.moments(img_BW)    
    scaled_huMoments = getHuMoments(moments)    

    # blob centroid
    centroid = ( int(moments['m10']/moments['m00']),
                 int(moments['m01']/moments['m00']) )

    # compile a feature vector
    features = np.append(scaled_huMoments, centroid)

    return features

if __name__ == "__main__":
    """
    Test feature extraction functions in main
    """

    # Building the file list
    fileList = loadFiles()

    for filename in fileList:
        print("[INFO] processing image: {}".format(filename))
        
        # Load image
        image = cv.imread(filename)
        
        imageMasked = prepareImage(image)
        
        largestContour = getLargestContour(imageMasked)
        
        contourExtremes = getContourExtremes(largestContour)
        
        convexityDefects = getConvexityDefects(largestContour)
        
        imageLargestContour = cv.drawContours(image.copy(), [largestContour], -1, (0, 255, 0), 2)

        # Draw convexity defects
        if convexityDefects is not None:
            for i in range(convexityDefects.shape[0]):
                s, e, f, d = convexityDefects[i]
                start = tuple(largestContour[s])
                end = tuple(largestContour[e])
                far = tuple(largestContour[f])
                
                # Draw a line between the start and end points
                cv.line(imageLargestContour, start, far, (0, 255, 255), 2)  # Yellow line for convexity defect
                cv.line(imageLargestContour, far, end, (0, 255, 255), 2)  # Yellow line for convexity defect
                
                # Draw a circle at the farthest defect point
                cv.circle(imageLargestContour, far, 5, (0, 0, 255), -1)  # Red circle for defect point
                
        cv.imshow('Contour features', imageLargestContour)
        
        # Extract features
        features = getFeatures(largestContour)
        
        print("[INFO] features: {}".format(features))
        
        
        # Go to the next image with any key press. Break loop if ESC is pressed
        k = cv.waitKey(0)
        if k == 27:
            break
        cv.destroyAllWindows()