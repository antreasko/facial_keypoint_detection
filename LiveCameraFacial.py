import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('/home/andreas/Desktop//Workfile/1/my_chinese_model_resnet.h5')
face_cascade = cv2.CascadeClassifier('/home/andreas/Desktop/Workfile/1/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # face_cascade = cv2.CascadeClassifier('/home/andreas/Desktop/Workfile/1/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    image_with_detections = np.copy(image)
    for (x,y,w,h) in faces:
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

        roi_color = gray[y:y + h, x:x + w]
        res = cv2.resize(roi_color, (96,96))
        reshape_img = np.reshape(res, (96,96,1)) / 255
        x = np.expand_dims(reshape_img, axis=0)
        imar = model.predict(x)
        for i in range(imar.shape[0]):
            orig_x,orig_y,orig_w,orig_h = faces[i]
    
    # denormalize points
            pts_x = imar[i][0::2] * orig_w/2 + orig_w/2 + orig_x 
            pts_y = imar[i][1::2] * orig_h/2 + orig_h/2 + orig_y
            for i in range(len(pts_x)):
                cv2.circle(image_with_detections, (pts_x[i], pts_y[i]), 1, (0, 0, 255), 2)  

    cv2.imshow("", image_with_detections)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()