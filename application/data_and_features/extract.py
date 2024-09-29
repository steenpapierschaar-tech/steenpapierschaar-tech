import os
import glob
import cv2 as cv
import numpy as np
from segment import maskBlueBG

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

    area = cv.contourArea(contour)

    contourLength = cv.arcLength(contour, True)

    ConvexHull  = cv.convexHull(contour, returnPoints=False)
    
    num_convex_hull_points = len(ConvexHull)

    sum_convex_hull_points = sum(list(map(sum, ConvexHull)))

    ConvexityDefects = getConvexityDefects(contour)

    num_convexity_defects = len(ConvexityDefects)

    sum_convexity_defects = sum(list(map(sum, ConvexityDefects)))

    features = np.array([contourLength, area, num_convex_hull_points, num_convexity_defects, sum_convex_hull_points, sum_convexity_defects])

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

    defect_depths = defects[:,-1]/256.0 if defects is not None else np.zeros((6,1))

    # select only the 6 largest depths
    defect_depths = np.flip(np.sort(defect_depths))[0:6]

    # compile a feature vector
    features = np.append(defect_depths, (area,perimeter))

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
    """ Test feature extraction functions"""

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    filename = os.path.join(__location__, 'demo.png')
        
    # load image and blur a bit to suppress noise
    img = cv.imread(filename)
    img = cv.blur(img,(3,3))

    # mask background
    img_BW = maskBlueBG(img)

    # perform a series of erosions and dilations to remove any small regions of noise
    img_BW = cv.erode(img_BW, None, iterations=2)
    img_BW = cv.dilate(img_BW, None, iterations=2)

    cv.imshow("Segmented image", img_BW)

    # extract features
    features = getBlobFeatures(img_BW)
    print("[INFO] blob features: {}".format(features))

    # find largest contour
    contour = getLargestContour(img_BW)

    # extract features
    features, defects = getContourFeatures(contour)
    print("[INFO] contour features: {}".format(features))

    # draw the outline of the object
    cv.drawContours(img, [contour], -1, (0, 255, 0), 2)

    area = cv.contourArea(contour)
    print("[TEST] contour area: {}".format(area))

    contourLength = cv.arcLength(contour, True)
    print("[TEST] contour length: {}".format(contourLength))

    ConvexHull  = cv.convexHull(contour, returnPoints=False)
    
    num_convex_hull_points = len(ConvexHull)
    print("[TEST] num_convex_hull_points: {}".format(num_convex_hull_points))

    sum_convex_hull_points = sum(list(map(sum, ConvexHull)))
    print("[TEST] sum_convex_hull_points: {}".format(sum_convex_hull_points))

    ConvexityDefects = getConvexityDefects(contour)
    
    num_convexity_defects = len(ConvexityDefects)
    print("[TEST] num_convexity_defects: {}".format(num_convexity_defects))

    sum_convexity_defects = sum(list(map(sum, ConvexityDefects)))
    print("[TEST] sum_convexity_defects: {}".format(sum_convexity_defects))

    features = np.array([contourLength, area, num_convex_hull_points, num_convexity_defects, sum_convex_hull_points, sum_convexity_defects])

    print("[TEST] contour features: {}".format(features))

    # point out hull defects
    if defects is not None:
        for s,e,f,d in defects:
            start = tuple(contour[s])
            end = tuple(contour[e])
            far = tuple(contour[f])
            cv.line(img,start,end,[0,255,255],2)
            cv.circle(img,far,5,[0,0,255],-1)

    # show result    
    cv.imshow("image", img)



    
