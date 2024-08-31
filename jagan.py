import cv2
import imutils

vs = cv2.VideoCapture(0)
Firstframe = None
area = 500

while True:
    _, img = vs.read()
    text = "Normal"
    
    # Resize the frame to a width of 500 pixels
    img = imutils.resize(img, width=500)
    
    # Convert to grayscale and apply Gaussian blur
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayimg, (21, 21), 0)
    
    # Initialize the first frame
    if Firstframe is None:
        Firstframe = gaussianImg
        continue
    
    # Compute the absolute difference between the first frame and current frame
    imgdiff = cv2.absdiff(Firstframe, gaussianImg)
    
    # Threshold the difference image, then dilate it to fill in holes
    _, threshImg = cv2.threshold(imgdiff, 20, 255, cv2.THRESH_BINARY)
    threshImg = cv2.dilate(threshImg, None, iterations=3)
    
    # Find contours
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        # If the contour is too small, ignore it
        if cv2.contourArea(c) < area:
            continue
        
        # Compute the bounding box for the contour and draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = "Moving Object detected"
    
    # Display the text on the frame
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow("camera", img)
    
    # Wait for the 'a' key to be pressed to exit
    key = cv2.waitKey(10)
    print(key)
    if key == ord('a'):
        break

# Release the video capture and close all windows
vs.release()
cv2.destroyAllWindows()

    
