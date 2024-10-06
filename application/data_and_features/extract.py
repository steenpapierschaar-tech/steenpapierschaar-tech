import os
import glob
import cv2 as cv
import numpy as np
from segment import maskWhiteBG
from dataset_loader import load_files

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

    # basic contour features
    area = cv.contourArea(contour)

    # get contour length
    contourLength = cv.arcLength(contour, True)

    # get convex hull
    ConvexHull  = cv.convexHull(contour, returnPoints=True)
    
    # Get convex hull length
    ConvexHullLength = cv.arcLength(ConvexHull, True)
    
    # initialize ConvexityDefects with a default value
    ConvexityDefects = 0.0
    
    # get the top 2 convexity defects
    defects = getConvexityDefects(contour)
    
    if defects is not None and defects.size > 0:
        # Select only the 2 largest defects
        ConvexityDefects = np.flip(np.sort(defects))[:2]
        # Get average of top 2 defects
        ConvexityDefects = np.mean(ConvexityDefects)
    
    compactness = contourLength/area

    # compile a feature vector
    features = np.array([area, contourLength, ConvexHullLength, ConvexityDefects,compactness])

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

    # Building the file list
    file_list = load_files()
    
    for filename in file_list:
        print("[INFO] processing image: {}".format(filename))
        
        # create a window to display images
        cv.namedWindow("Extracted features")
            
        # load image and blur a bit to suppress noise
        img = cv.imread(filename)
        img_label = img.copy()
        img_label = cv.putText(img_label, "1. Original", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # mask background
        # perform a series of erosions and dilations to remove any small regions of noise
        img_mask = img.copy()
        img_mask = maskWhiteBG(img)
        img_mask = cv.erode(img_mask, None, iterations=2)
        img_mask = cv.dilate(img_mask, None, iterations=2)
        img_mask_label = cv.putText(img_mask, "2. Mask", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # extract features
        features = getBlobFeatures(img_mask)

        # find largest contour. Draw it on the image
        img_contour = img.copy()
        contour = getLargestContour(img_mask)
        img_contour = cv.drawContours(img_contour, [contour], -1, (0, 255, 0), 2)
        img_contour_label = cv.putText(img_contour, "3. Contour", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # extract features
        features = getContourFeatures(contour)

        # Contour Area. Fill area in picture
        area = cv.contourArea(contour)
        img_area = img.copy()
        img_area = cv.drawContours(img_area, [contour], -1, (0, 255, 0), -1)
        img_area_label = cv.putText(img_area, "4. Area: {:.0f}".format(area), (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Contour Length
        contourLength = cv.arcLength(contour, True)
        img_contourLength = img.copy()
        img_contourLength_label = cv.putText(img_contourLength, "5. Length: {:.0f}".format(contourLength), (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Get the convex hull of the contour. Draw the hull on the image
        convexhull = cv.convexHull(contour, returnPoints=False)
        convexhull_points = contour[convexhull[:, 0]]
        img_convexhull = img.copy()
        img_convexhull = cv.drawContours(img_convexhull, [convexhull_points], -1, (0, 0, 255), 2)
        img_convexhull_label = cv.putText(img_convexhull, "6. Convexity Hull", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Get the convexity defects of the contour
        features, defects = getContourFeatures(contour)
        
        if defects is not None:
            img_convexityDefects = img.copy()
            img_convexityDefects = cv.drawContours(img_convexityDefects, [contour], -1, (0, 255, 0), 2)
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i]  # Unpack the start, end, and farthest point indices, and ignore the depth
                start = tuple(contour[s].ravel())  # Flatten the array to get x, y coordinates
                end = tuple(contour[e].ravel())    # Same for the end point
                far = tuple(contour[f].ravel())    # Same for the farthest point
                cv.circle(img_convexityDefects, far, 5, (0, 0, 255), -1)
                cv.line(img_convexityDefects, start, end, (0, 0, 255), 2)

            img_convexityDefects_label = cv.putText(img_convexityDefects, "7. Convexity Defects", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


        # Resize all images to the same dimensions
        height, width = img.shape[:2]
        img_mask_resized = cv.resize(img_mask_label, (width, height))
        img_contour_resized = cv.resize(img_contour_label, (width, height))
        img_area_resized = cv.resize(img_area_label, (width, height))
        img_contourLength_resized = cv.resize(img_contourLength_label, (width, height))
        img_convexhull_resized = cv.resize(img_convexhull_label, (width, height))
        img_convexityDefects_resized = cv.resize(img_convexityDefects_label, (width, height))

        # Ensure all images have the same type
        img_mask_resized = cv.cvtColor(img_mask_resized, cv.COLOR_GRAY2BGR)

        # Show the mask and the masked image combined
        combined_result_row_a = cv.hconcat([img_label, img_mask_resized, img_contour_resized, img_convexityDefects_resized])
        combined_result_row_b = cv.hconcat([img_area_resized, img_contourLength_resized, img_convexhull_resized, img_convexityDefects_resized])
        
        combined_result = cv.vconcat([combined_result_row_a, combined_result_row_b])
        
        cv.imshow("Extracted features", combined_result)
        
        # Print measurements
        features = getFeatures(contour)
        # Access elements by index
        area = features[0]
        contourLength = features[1]
        ConvexHullLength = features[2]
        ConvexityDefects = features[3]
        compactness = features[4]
        
        print("Area: {:.0f}".format(area), 
              "Contour Length: {:.0f}".format(contourLength), 
              "Convex Hull Length: {:.0f}".format(ConvexHullLength), 
              "Convexity Defects: {:.0f}".format(ConvexityDefects), 
              "Compactness: {:.2f}".format(compactness))
        
        # wait for a key press
        cv.waitKey(0)