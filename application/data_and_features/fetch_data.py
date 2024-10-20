import os
import glob
import cv2 as cv
import numpy as np
from segment import maskWhiteBG
from extract import getLargestContour, getFeatures
from sklearn.utils import Bunch
from pipeline import prepimg, testGrabCut

def fetch_data(data_path):
    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.png'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # intitialize data matrix with correct number of features
    feature_names = ['area', 'contourLength', 'ConvexHullLength', 'ConvexityDefects','compactness']

    data = np.empty((0,len(feature_names)), float)
    target = []
  
    # loop over the image paths
    for filename in file_list: #[::10]:
        # load image and blur a bit to suppress noise
        img = cv.imread(filename)
        #img_foreground = prepimg(img)
        img_foreground = testGrabCut(img)


        # find largest contour
        contour = getLargestContour(img_foreground)

        # extract features from contour
        #features = getSimpleContourFeatures(contour)
        features = getFeatures(contour)
        print("[INFO] contour features: {}".format(features))

        # extract label from folder name and stor
        label = filename.split(os.path.sep)[-2]
        target.append(label)

        # append features to data matrix
        data = np.append(data, np.array([features]), axis=0)

        # draw outline, show image, and wait a bit
        cv.drawContours(img, [contour], -1, (0, 255, 0), 2)        
        cv.imshow("image", img)
        #k = cv.waitKey(1) & 0xFF

        cv.waitKey(1)
        # if the `q` key or ESC was pressed, break from the loop
        #if k == ord("q") or k == 27:
        #    break

    unique_targets = np.unique(target)
    print("[INFO] targets found: {}".format(unique_targets))

    dataset = Bunch(data = data,
                    target = target,
                    unique_targets = unique_targets,
                    feature_names = feature_names)

    return dataset


# snippet:
##    scatter = ax0.scatter(trainX[:,0], trainX[:,1], c=coded_labels)
    # produce a legend with the unique colors from the scatter
##    handles, _ = scatter.legend_elements(prop="colors")
##    legend0 = ax0.legend(handles, unique_labels,
##                        loc="upper left", title="Classes")
##    ax0.add_artist(legend0)

if __name__ == "__main__":
    """ Test fetching function"""
    data_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'photo_dataset')
    
    gestures = fetch_data(data_path)

    print(dir(gestures))
    print(gestures.unique_targets)