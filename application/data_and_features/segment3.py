import cv2 as cv
import numpy as np
import time
from dataset_loader import load_files

# Function to handle trackbar changes
def update_threshold(val):
    pass

# Setup windows and trackbars
cv.namedWindow('HSV Thresholding')
cv.namedWindow('HSL Thresholding')
cv.namedWindow('YCbCr Thresholding')

# Create trackbars for each threshold value (initial values set to 0)
cv.createTrackbar('HSV H Min', 'HSV Thresholding', 0, 255, update_threshold)
cv.createTrackbar('HSV H Max', 'HSV Thresholding', 255, 255, update_threshold)
cv.createTrackbar('HSV S Min', 'HSV Thresholding', 0, 255, update_threshold)
cv.createTrackbar('HSV S Max', 'HSV Thresholding', 255, 255, update_threshold)
cv.createTrackbar('HSV V Min', 'HSV Thresholding', 0, 255, update_threshold)
cv.createTrackbar('HSV V Max', 'HSV Thresholding', 255, 255, update_threshold)

cv.createTrackbar('HSL H Min', 'HSL Thresholding', 0, 255, update_threshold)
cv.createTrackbar('HSL H Max', 'HSL Thresholding', 255, 255, update_threshold)
cv.createTrackbar('HSL L Min', 'HSL Thresholding', 0, 255, update_threshold)
cv.createTrackbar('HSL L Max', 'HSL Thresholding', 255, 255, update_threshold)
cv.createTrackbar('HSL S Min', 'HSL Thresholding', 0, 255, update_threshold)
cv.createTrackbar('HSL S Max', 'HSL Thresholding', 255, 255, update_threshold)

cv.createTrackbar('YCbCr Y Min', 'YCbCr Thresholding', 0, 255, update_threshold)
cv.createTrackbar('YCbCr Y Max', 'YCbCr Thresholding', 255, 255, update_threshold)
cv.createTrackbar('YCbCr Cb Min', 'YCbCr Thresholding', 0, 255, update_threshold)
cv.createTrackbar('YCbCr Cb Max', 'YCbCr Thresholding', 255, 255, update_threshold)
cv.createTrackbar('YCbCr Cr Min', 'YCbCr Thresholding', 0, 255, update_threshold)
cv.createTrackbar('YCbCr Cr Max', 'YCbCr Thresholding', 255, 255, update_threshold)

# Variable to store the last time GrabCut was executed
last_grabcut_time = 0
grabcut_interval = 2  # 2 seconds interval for GrabCut execution

# Load all dataset image files
file_list = load_files()

for filename in file_list:
    print("[INFO] Processing image: {}".format(filename))
    
    # Load the image
    img = cv.imread(filename)
    img_original = img.copy()

    # Boost contrast of image
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Blur the image to suppress noise
    img_blur = cv.blur(img, (6, 6))

    # Create a HSV, HSL, and YCbCr copy of the image
    img_hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
    img_hsl = cv.cvtColor(img_blur, cv.COLOR_BGR2HLS)
    img_ycbcr = cv.cvtColor(img_blur, cv.COLOR_BGR2YCrCb)

    # Main loop to update images based on trackbar position
    while True:
        
        # Get current positions of trackbars for threshold values
        hsv_h_min = cv.getTrackbarPos('HSV H Min', 'HSV Thresholding')
        hsv_h_max = cv.getTrackbarPos('HSV H Max', 'HSV Thresholding')
        hsv_s_min = cv.getTrackbarPos('HSV S Min', 'HSV Thresholding')
        hsv_s_max = cv.getTrackbarPos('HSV S Max', 'HSV Thresholding')
        hsv_v_min = cv.getTrackbarPos('HSV V Min', 'HSV Thresholding')
        hsv_v_max = cv.getTrackbarPos('HSV V Max', 'HSV Thresholding')
        
        hsl_h_min = cv.getTrackbarPos('HSL H Min', 'HSL Thresholding')
        hsl_h_max = cv.getTrackbarPos('HSL H Max', 'HSL Thresholding')
        hsl_l_min = cv.getTrackbarPos('HSL L Min', 'HSL Thresholding')
        hsl_l_max = cv.getTrackbarPos('HSL L Max', 'HSL Thresholding')
        hsl_s_min = cv.getTrackbarPos('HSL S Min', 'HSL Thresholding')
        hsl_s_max = cv.getTrackbarPos('HSL S Max', 'HSL Thresholding')
        
        ycbcr_y_min = cv.getTrackbarPos('YCbCr Y Min', 'YCbCr Thresholding')
        ycbcr_y_max = cv.getTrackbarPos('YCbCr Y Max', 'YCbCr Thresholding')
        ycbcr_cb_min = cv.getTrackbarPos('YCbCr Cb Min', 'YCbCr Thresholding')
        ycbcr_cb_max = cv.getTrackbarPos('YCbCr Cb Max', 'YCbCr Thresholding')
        ycbcr_cr_min = cv.getTrackbarPos('YCbCr Cr Min', 'YCbCr Thresholding')
        ycbcr_cr_max = cv.getTrackbarPos('YCbCr Cr Max', 'YCbCr Thresholding')

        # Apply custom thresholding using inRange for HSV
        hsv_lower = np.array([hsv_h_min, hsv_s_min, hsv_v_min], dtype=np.uint8)
        hsv_upper = np.array([hsv_h_max, hsv_s_max, hsv_v_max], dtype=np.uint8)
        hsv_mask = cv.inRange(img_hsv, hsv_lower, hsv_upper)
        
        # Apply custom thresholding using inRange for HSL
        hsl_lower = np.array([hsl_h_min, hsl_l_min, hsl_s_min], dtype=np.uint8)
        hsl_upper = np.array([hsl_h_max, hsl_l_max, hsl_s_max], dtype=np.uint8)
        hsl_mask = cv.inRange(img_hsl, hsl_lower, hsl_upper)

        # Apply custom thresholding using inRange for YCbCr
        ycbcr_lower = np.array([ycbcr_y_min, ycbcr_cb_min, ycbcr_cr_min], dtype=np.uint8)
        ycbcr_upper = np.array([ycbcr_y_max, ycbcr_cb_max, ycbcr_cr_max], dtype=np.uint8)
        ycbcr_mask = cv.inRange(img_ycbcr, ycbcr_lower, ycbcr_upper)

        # Combine the masks
        combined_mask_hsv = hsv_mask
        combined_mask_hsl = hsl_mask
        combined_mask_ycbcr = ycbcr_mask

        # Invert the combined masks
        binary_inv_hsv = cv.bitwise_not(combined_mask_hsv)
        binary_inv_hsl = cv.bitwise_not(combined_mask_hsl)
        binary_inv_ycbcr = cv.bitwise_not(combined_mask_ycbcr)

        # Create masks for GrabCut based on inverted thresholds
        mask_hsv = np.where(binary_inv_hsv > 0, 1, 0).astype('uint8')
        mask_hsl = np.where(binary_inv_hsl > 0, 1, 0).astype('uint8')
        mask_ycbcr = np.where(binary_inv_ycbcr > 0, 1, 0).astype('uint8')

        # Mark sure background as 0 and sure foreground as 1 to help GrabCut
        mask_hsv[0:5, :] = cv.GC_BGD  # sure background (top 5 rows)
        mask_hsv[-5:, :] = cv.GC_FGD  # sure foreground (bottom 5 rows)
        mask_hsl[0:5, :] = cv.GC_BGD
        mask_hsl[-5:, :] = cv.GC_FGD
        mask_ycbcr[0:5, :] = cv.GC_BGD
        mask_ycbcr[-5:, :] = cv.GC_FGD

        # Check if 2 seconds have passed since the last GrabCut execution
        current_time = time.time()
        if current_time - last_grabcut_time >= grabcut_interval:
            # Create temporary arrays used by the GrabCut algorithm
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            # Create img copy for each color space
            img_hsv_copy = img_original.copy()
            img_hsl_copy = img_original.copy()
            img_ycbcr_copy = img_original.copy()

            # Apply GrabCut with masks
            cv.grabCut(img_hsv_copy, mask_hsv, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
            cv.grabCut(img_hsl_copy, mask_hsl, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
            cv.grabCut(img_ycbcr_copy, mask_ycbcr, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

            # Update the last time GrabCut was executed
            last_grabcut_time = current_time

        # Modify the masks such that background is 0 and foreground is 1
        mask2_hsv = np.where((mask_hsv == 2) | (mask_hsv == 0), 0, 1).astype('uint8')
        mask2_hsl = np.where((mask_hsl == 2) | (mask_hsl == 0), 0, 1).astype('uint8')
        mask2_ycbcr = np.where((mask_ycbcr == 2) | (mask_ycbcr == 0), 0, 1).astype('uint8')

        # Mask the image to only show the foreground (the hand)
        hand_segmented_hsv = img_original * mask2_hsv[:, :, np.newaxis]
        hand_segmented_hsl = img_original * mask2_hsl[:, :, np.newaxis]
        hand_segmented_ycbcr = img_original * mask2_ycbcr[:, :, np.newaxis]

        # Apply morphological operations to clean up the mask
        kernel = np.ones((9, 9), np.uint8)
        mask2_hsv = cv.morphologyEx(mask2_hsv, cv.MORPH_CLOSE, kernel)
        mask2_hsl = cv.morphologyEx(mask2_hsl, cv.MORPH_CLOSE, kernel)
        mask2_ycbcr = cv.morphologyEx(mask2_ycbcr, cv.MORPH_CLOSE, kernel)
        
        # Mask the image again with the refined mask
        hand_segmented_hsv_refined = img_original * mask2_hsv[:, :, np.newaxis]
        hand_segmented_hsl_refined = img_original * mask2_hsl[:, :, np.newaxis]
        hand_segmented_ycbcr_refined = img_original * mask2_ycbcr[:, :, np.newaxis]
        
        # Draw contours around the hand
        contours_hsv, _ = cv.findContours(mask2_hsv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_hsl, _ = cv.findContours(mask2_hsl, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_ycbcr, _ = cv.findContours(mask2_ycbcr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the image to draw the contours
        img_contours_hsv = img_original.copy()
        img_contours_hsl = img_original.copy()
        img_contours_ycbcr = img_original.copy()
        
        # Draw the contours
        cv.drawContours(img_contours_hsv, contours_hsv, -1, (0, 255, 0), 2)
        cv.drawContours(img_contours_hsl, contours_hsl, -1, (0, 255, 0), 2)
        cv.drawContours(img_contours_ycbcr, contours_ycbcr, -1, (0, 255, 0), 2)
        
        # Display the original and segmented image
        hsv_display = cv.hconcat([img, img_contours_hsv])
        hsv_segmented_display = cv.hconcat([hand_segmented_hsv, hand_segmented_hsv_refined])
        cv.imshow('HSV Thresholding', cv.vconcat([hsv_display, hsv_segmented_display]))

        hsl_display = cv.hconcat([img, img_contours_hsl])
        hsl_segmented_display = cv.hconcat([hand_segmented_hsl, hand_segmented_hsl_refined])
        cv.imshow('HSL Thresholding', cv.vconcat([hsl_display, hsl_segmented_display]))

        ycbcr_display = cv.hconcat([img, img_contours_ycbcr])
        ycbcr_segmented_display = cv.hconcat([hand_segmented_ycbcr, hand_segmented_ycbcr_refined])
        cv.imshow('YCbCr Thresholding', cv.vconcat([ycbcr_display, ycbcr_segmented_display]))

        # Exit the loop on pressing 'q' or 'ESC'
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

# Destroy all windows after exiting
cv.destroyAllWindows()
