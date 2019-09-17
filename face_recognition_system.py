import os
import cv2
from PIL import Image
import numpy as np
import pickle


def data(Id):

    cam = cv2.VideoCapture(0)
    detector=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

    path = os.getcwd()
    path = path + "/images/"

    if Id in os.listdir(path):
        img_path = path + str(Id)
    else:
        img_path = path + str(Id)
        os.mkdir(img_path)

    sampleNum = len(os.listdir(img_path))

    while(True):
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x,y,w,h) in faces:

            sampleNum=sampleNum+1
            cv2.imwrite(img_path +'/'+ str(sampleNum) + ".jpg", frame[y-10:y+h+10,x-10:x+w+10])

        cv2.imshow('frame',frame)

        if sampleNum==300:
            print("These data are enough for Training.")
            break

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()



def train():

    detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    path = os.getcwd()
    path = path + "/images/"

    for root, dirs, files in os.walk(path):

        for file in files:

            if file.endswith("JPG") or file.endswith("jpg") or file.endswith("png") or file.endswith("PNG"):
                img_path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ","_").lower()

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                
                id_ = label_ids[label]
                pil_img = Image.open(img_path).convert("L")
                size = (480,480)
                final_img = pil_img.resize(size, Image.ANTIALIAS)
                img_array = np.array(final_img, "uint8")
                faces = detector.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = img_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")



def recognize():
    cam = cv2.VideoCapture(0)
    detector=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read("trainner.yml")
    
    labels={"person_name": 1}
    with open("labels.pickle", 'rb') as f:
        orig_labels = pickle.load(f)
        labels = {v:k for k,v in orig_labels.items()}

    while True:
        ret, frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            id_,conf = recognizer.predict(roi_gray)

            if conf>=45:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255,255,255)
                stroke = 2
                acc = str(name)
                cv2.putText(frame, acc, (x,y-10), font, 1, color, stroke, cv2.LINE_AA)

            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = "Unknown"
                color = (255,255,255)
                stroke = 2
                cv2.putText(frame, name, (x,y-10), font, 1, color, stroke, cv2.LINE_AA)

            img_item = "img.png"
            cv2.imwrite(img_item, roi_color)
            
            color = (255, 0, 0) 
            stroke = 2
            width = x+w 
            height = y+h 
            cv2.rectangle(frame, (x, y), (width, height), color, stroke)

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

