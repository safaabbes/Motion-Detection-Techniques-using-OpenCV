import cv2
import numpy as np

frames=[]
MAX_FRAMES = 1000
N = 30
THRESH = 50
ASSIGN_VALUE = 255 ## Value to assign the pixel if the threshold is met

webcam = False

if webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("material/Video.mp4")
    
for t in range(MAX_FRAMES):
    #Capture frame by frame
    ret, frame = cap.read()
    #When Video reaches its end
    if not ret:
        break
    #Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
    #Append to list of frames
    frames.append(frame_gray)
    
    if t >= N:
        #D(N) = || I(t) - I(t+N) || = || I(t-N) - I(t) ||
        diff = cv2.absdiff(frames[t-N], frames[t])
        #Mask Thresholding
        threshold_method = cv2.THRESH_BINARY
        ret, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, threshold_method)
        #Display the Motion Mask
        cv2.imshow('Motion Mask', motion_mask)
    
    #Display the resulting frame
    cv2.imshow('Frame', frame)
    #Wait and exit if q is pressed
    if cv2.waitKey(1) == ord('q') or not ret:
        break

#When everything is finished, we release the capture
cap.release()
cv2.destroyAllWindows()
        

    

