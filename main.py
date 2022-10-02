import cv2
from deepface import DeepFace
import numpy as np
import webbrowser

print(">>> A python program to filter songs according to their current mood")

print("[!] Press 'y' to open playlist in browser")
print("[!] Press 'q' to exit")

# ref xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# YT Url Generator
def genUrl(result):
    url = "https://www.youtube.com/results?search_query=" + result + "+mood+songs"
    print("[#] Openning " + url)
    webbrowser.open_new(url)

#Webcam
video=cv2.VideoCapture(0)

while video.isOpened:
    ret, frame =video.read()
    
    if ret == True:
        # Gray Scale
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Detection Box
        for x,y,w,h in face:
          img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) 
      
        # Detect Emotion
        try:
            emotion = DeepFace.analyze(frame)
            result = emotion['dominant_emotion']
            print(result)
        except:
            print("no face")  

        cv2.imshow('video', frame)
        
        key=cv2.waitKey(1) 
        if key==ord('q'):
            break
        elif key==ord('y'):
            genUrl(result)
            break
            
    video.release()
