import cv2
import numpy as np

# Init camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'

file_name = input("Enter the name of the person: ")
while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    if len(faces) == 0:  # Verifica se há rostos detectados
        continue

    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

    # pick the last face (because it is the largest face acc to area f[2]*f[3])
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Extract (Crop out required face)
        offset = 10
        face_section = frame[y-offset: y+h+offset, x-offset: x+w+offset] 
        face_section = cv2.resize(face_section, (100, 100))

        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Frame", frame)
    cv2.imshow("Face Section", face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
# Convert our face list array into a np array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Save this data
np.save(dataset_path+file_name+'.npy', face_data)
print("Data collected with sucess!")

cap.release()
cv2.destroyAllWindows()