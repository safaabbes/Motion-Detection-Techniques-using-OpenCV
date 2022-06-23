import numpy as np
import cv2

background = None
MAX_FRAMES = 1000
THRESH = 60
ASSIGN_VALUE = 255
ALPHA = 0.1

def update_background(current_frame, prev_bg, alpha):
    bg = alpha * current_frame + (1 - alpha) * prev_bg
    bg = np.uint8(bg)  
    return bg

cap = cv2.VideoCapture(0)
    
for t in range(MAX_FRAMES):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)      
    if t == 0:
        # Train background with first frame
        background = frame_gray
    else:
        # Background subtraction
        diff = cv2.absdiff(background, frame_gray)
        # Mask thresholding
        ret, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, cv2.THRESH_BINARY)
        # Update background
        background = update_background(frame_gray, background, alpha = ALPHA)
        # Display the motion mask and background
        cv2.imshow('Motion mask', motion_mask)
        cv2.imshow('Background', background)
        # Exit 
        if cv2.waitKey(1) == ord('e') or not ret:
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



