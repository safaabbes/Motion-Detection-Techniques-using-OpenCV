import cv2

MAX_FRAMES = 1000
LEARNING_RATE = -1   
fgbg = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)

for t in range(MAX_FRAMES):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #Apply MOG 
    motion_mask = fgbg.apply(frame, LEARNING_RATE)
    #Get background
    background = fgbg.getBackgroundImage()
    # Display the motion mask and background
    cv2.imshow('background', background)
    cv2.imshow('Motion Mask', motion_mask)
    # Exit
    if cv2.waitKey(1) == ord('e') or not ret:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
