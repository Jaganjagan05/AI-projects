import cv2
alg=r"C:/Users/majag/Downloads/haarcascade_frontalface_default (1).xml"#algorithem

haar_cascade=cv2.CascadeClassifier(alg)
video_Link=r"C:\Users\majag\Downloads\WhatsApp Video 2024-09-01 at 11.35.45_1da40ca8.mp4"

video=cv2.VideoCapture(video_Link)#video

while True:
    _,img=video.read()
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#img-gray
    face=haar_cascade.detectMultiScale(grayimg,1.3,4)#get x,y,w,h
    
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,200),3)#draw rectangle
        
    cv2.imshow("FaceDetection",img)
    key=cv2.waitKey(10)
    print(key)
    if key==ord('s'):#stop to click s
        break
    
video.release()
cv2.destroyAllWindows()
