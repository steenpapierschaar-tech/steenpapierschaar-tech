##Create virtual envoirement
#python -m venv --system-site-packages venv

##Activate virtual envoirement
#source venv/bin/activate

##Install libraries using pip
#pip install numpy scipy scikit-learn imutils opencv-python-headless matplotlib

##Run scripts 
#python script_xyz.py
#######################################################################################################


import os
import time
import cv2 as cv
import numpy as np
from picamera2 import Picamera2
#from imutils.video import VideoStream

data_path = 'data'
mappings = {"i": "ignore",
            "r": "rock",
            "p": "paper",
            "s": "scissors",
            "h": "hang_loose"}
frame_size = (320,240) 

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": frame_size})
picam2.configure(camera_config)
picam2.start()

# map the keys to their ordinal numbers
kMappings = {}
for key in mappings.keys():
    kMappings[ord(key)] = mappings[key]

#vs = VideoStream(src=0, usePiCamera=False, resolution=frame_size).start()

time.sleep(1.0)

print(mappings)

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream
    #frame = vs.read()
    frame = picam2.capture_array()
    frame = cv.cvtColor( frame, cv.COLOR_BGR2RGB )

    if frame is None:
        continue

    cv.imshow("Frame", frame)
    cv.resizeWindow('Frame', 320,240)

    k = cv.waitKey(1) & 0xFF

    #print("Key {} is pressed", k)
    
    # if the `q` key or ESC was pressed, break from the loop
    if k == ord("q") or k == 27:
        break
    # otherwise, check to see if a key was pressed that we are
    # interested in capturing
    elif k in kMappings.keys():
        # construct the path to the label subdirectory
        p = os.path.sep.join([data_path, kMappings[k]])
        if not os.path.exists(p):
            os.makedirs(p)
        # construct the path to the output image
        p = os.path.sep.join([p, "{}.png".format(int(time.time_ns()))])
        print("[INFO] saving frame: {}".format(p))
        cv.imwrite(p, frame)

# Clean up
cv.destroyAllWindows()
picam2.stop()
