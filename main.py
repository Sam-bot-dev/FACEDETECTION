import cv2 
face_cap=cv2.CascadeClassifier("haar_face.xml")
vid_capture = cv2.VideoCapture(0)

while True:
    ret, video_data = vid_capture.read()
    col= cv2.cvtColor(video_data,cv2.COLOR_BGRA2GRAY)
    faces= face_cap.detectMultiScale(col,scaleFactor=1.1,minNeighbors=3,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(255,255,255),2)
    cv2.imshow("live",video_data)
    if cv2.waitKey(10) == ord("a"):
        break
vid_capture.release()
