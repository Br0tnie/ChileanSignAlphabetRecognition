import cv2
import torch
from ultralytics import YOLO
import numpy as np
import math
import cvzone
from datetime import timedelta, datetime
import collections
from tkinter import *



predictions=[]
word=[]
# Load the YOLOv8 model
model = YOLO('runs/detect/train13/weights/best.pt')
model.model.to(torch.device('cpu')) #SOLO SI NO HAY CPU

className = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
             'U', 'V', 'W', 'X', 'Y', 'Z']

# Créez une nouvelle fenêtre Tkinter
root = Tk()
root.title("Interpreter")

def helloCallBack():
   word.pop()


# Créez une étiquette pour afficher la prédiction
buton = Button(root, text ="Borrar letra",font=("Arial", 20),justify=CENTER, command = helloCallBack)
buton.pack()
prediction_label = Label(root, text="No hay letra",padx=1,pady=1, justify=CENTER, font=("Arial", 40))
prediction_label.pack()
word_label = Label(root, text="Palabra",padx=1,pady=1, justify=CENTER, font=("Arial", 40))
word_label.pack()

# Open the video file
cap = cv2.VideoCapture(-1) #Revisar existencia de cámara
if (cap.isOpened()== False):
    print("Error opening video stream or file")

fileName = "salida.mp4"
codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frameRate = 20.0
resolution = (640,480)

videoOut = cv2.VideoWriter(fileName,codec, frameRate, resolution)
window = timedelta(seconds=2)
initTime=datetime.now()
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.5)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                #print("predicted class",className[cls])
                predictions.append(className[cls])

        if datetime.now()-initTime >= window:
            if predictions:
                counter = collections.Counter(predictions)
                most_common = counter.most_common(1)[0][0]
                word.append(most_common)
                prediction_label.config(text=f"Letra : {most_common}")
                word_label.config(text=f"{''.join(word)}")
            else:
                prediction_label.config(text=f"No hay letra")
                word_label.config(text=f"{''.join(word)}")
            predictions.clear()
            initTime=datetime.now()
        root.update()
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        videoOut.write(annotated_frame)
        cv2.imshow("YOLOv8 Inference", annotated_frame)
    

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
root.destroy()
cap.release()
cv2.destroyAllWindows()
