import cv2
import numpy as np
import matplotlib.pyplot as plt



# Load the image
image_path = "1.bmp"
image = cv2.imread(image_path)

# Convert the image to grayscale
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
img = hsv[:,:,0] 


# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(img, (21, 21), 0)

# Threshold the image to create a binary image
ret, threshold = cv2.threshold(blurred, 80,100 , cv2.THRESH_BINARY_INV)

#histr = cv2.calcHist([blurred],[0],None,[256],[0,256])
#plt.plot(histr)
#plt.show()

#cv2.imshow("threshold", threshold)
# Find contours in the thresholded image
contours, ret = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the convex hull for each contour found
for contour in contours:
    hull = cv2.convexHull(contour)
    cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)   


# Display the original image with the convex hull
cv2.imshow("Convex Hull", image)
cv2.waitKey(0)
cv2.destroyAllWindows()