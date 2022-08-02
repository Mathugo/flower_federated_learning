import cv2, os
import uuid
filename = str(uuid.uuid4())

axis_cam = "http://orange:Orange*@192.168.0.20/axis-cgi/mjpg/video.cgi"
#import numpy as np
#cam = 0 # Use  local webcam.
folder = "dataset"

cap = cv2.VideoCapture(axis_cam)
if not cap:
    print("!!! Failed VideoCapture: invalid parameter!")

while(True):
    # Capture frame-by-frame
    ret, current_frame = cap.read()
    if type(current_frame) == type(None):
        print("!!! Couldn't read frame!")
        break

    # Display the resulting frame
    cv2.imshow('frame',current_frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        filename = os.path.join(folder, str(uuid.uuid4())+".jpg")
        cv2.imwrite(filename, current_frame)
        print("Screened")

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture
cap.release()
cv2.destroyAllWindows()