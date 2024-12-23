import numpy as np 
import cv2 
import os 
import matplotlib.pyplot as plt

def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

cap = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier("./frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("./Nose18x15.xml")

# Carrega a imagem que será sobreposta aos olhos
overlay = cv2.imread("./glasses.png", -1)
overlay_nose = cv2.imread("./mustache.png", -1)

dataset_path = './data/'
face_data = []
labels = []
class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta os olhos
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # Detecta o nariz
    noses = nose_cascade.detectMultiScale(gray, 1.3, 5)
    
    for eye, nose in zip(eyes, noses):
        
        ex, ey, ew, eh = eye
        exn, eyn, ewn, ehn = nose
        eyn = eyn + 30 
        
        # Redimensiona a imagem de sobreposição para o tamanho da região dos olhos
        overlay_resized = cv2.resize(overlay, (ew, eh))

        # Redimensiona a imagem de sobreposição para o tamanho da região do nariz
        overlay_resized_nose = cv2.resize(overlay_nose, (ewn, ehn))

        # Região onde a imagem será sobreposta
        roi = frame[ey:ey+eh, ex:ex+ew]

        roi_nose = frame[eyn:eyn+ehn, exn:exn+ewn]
        
        if overlay_resized.shape[2] == 4:
            alpha = overlay_resized[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
        
        if overlay_resized_nose.shape[2] == 4:
            alpha_nose = overlay_resized_nose[:, :, 3] / 255.0
            for c in range(3):
                roi_nose[:, :, c] = roi_nose[:, :, c] * (1 - alpha_nose) + overlay_resized_nose[:, :, c] * alpha_nose
    
    cv2.imshow("Eyes", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



