import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "1.bmp"
image = cv2.imread(image_path)

# Create a mask initialized to probable background
mask = np.zeros(image.shape[:2], np.uint8)

# Define the background and foreground models (needed for GrabCut)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Define a rectangle containing the foreground object (the hand)
# You can adjust these coordinates if necessary
rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

# Apply GrabCut algorithm
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask such that all sure and probable foreground pixels are set to 1
# and the rest are set to 0
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Multiply the input image with the new mask to remove the background
result = image * mask2[:, :, np.newaxis]

# Display the result
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()