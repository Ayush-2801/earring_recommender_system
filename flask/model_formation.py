import dlib
import math
import cv2
import numpy as np
from imutils import face_utils
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

model_path = "shape_predictor_81_face_landmarks.dat"
model = dlib.shape_predictor(model_path)
detector = dlib.get_frontal_face_detector()

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def angle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    vector1 = (x1 - x2, y1 - y2)
    vector2 = (x3 - x2, y3 - y2)
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    cosine = dot_product / (magnitude1 * magnitude2)
    angle_rad = math.acos(cosine)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def midpoint(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    midpoint = [midpoint_x, midpoint_y]
    return midpoint

data_pth = "FaceShape Dataset/training_set"
faces = []
labels = []
for shape in os.listdir(data_pth):
    if(shape == "desktop.ini"):
        continue
    else:
        k = shape
        shape_pth = os.path.join(data_pth, shape)
        for img in os.listdir(shape_pth):
            img_pth = os.path.join(shape_pth, img)
            pic = cv2.imread(img_pth)
            if pic is None:
                # print("Error: Failed to load image or image file does not exist.")
                continue
            else:
                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            dets = detector(pic)
            for face in dets:
                shape = model(pic, face)
                coords = face_utils.shape_to_np(shape)
            l3 = coords[3]
            l15 = coords[15]
            l70 = coords[70]
            l76 = coords[76]
            l80 = coords[80]
            l73 = coords[73]
            l9 = coords[9]
            l13 = coords[13]
            l5 = coords[5]
            l7 = coords[7]
            l11 = coords[11]
            l8 = coords[8]
            l10 = coords[10]
            d1 = distance(l3, l15)
            d2 = distance(l76, l80)
            d3 = distance(midpoint(l70, l73), l9)
            d4 = distance(l9, l13)
            d5 = distance(l5, l13)
            d6 = distance(l7, l11)
            d7 = distance(l8, l10)
            DD = d1 + d2 + d3 + d4 + d5 + d6 + d7
            D1 = d1/DD
            D2 = d2/DD
            D3 = d3/DD
            D4 = d4/DD
            D5 = d5/DD
            D6 = d6/DD
            D7 = d7/DD
            R1 = D2/D1
            R2 = D1/D3
            R3 = D2/D3
            R4 = D1/D5
            R5 = D6/D5
            R6 = D4/D6
            R7 = D6/D1
            R8 = D5/D2
            R9 = D4/D5
            R10 = D7/D6
            A1 = angle(midpoint(l70, l73), l9, l11)
            A2 = angle(midpoint(l70, l73), l9, l13)
            A3 = angle(l3, l15, l13)
            features = []
            features.append(R1)
            features.append(R2)
            features.append(R3)
            features.append(R4)
            features.append(R7)
            features.append(R8)
            features.append(R10)
            features.append(D1)
            features.append(D2)
            features.append(D3)
            features.append(D5)
            features.append(D6)
            features.append(A1)
            features.append(A2)
            features = np.array(features)
            if k == "Heart":
                labels.append(0)
            elif k == "Oblong":
                labels.append(1)
            elif k == "Oval":
                labels.append(2)
            elif k == "Round":
                labels.append(3)
            elif k == "Square":
                labels.append(4)
            else: 
                print('ERROR')
                break
            faces.append(features)

f = open('data/df.pickle', 'wb')
pickle.dump({'data': faces, 'labels': labels}, f)
f.close()

df = pickle.load(open('data/df.pickle', 'rb'))
faces = df['data']
labels = df['labels']

X_train = np.array(faces)
y_train = np.array(labels)
X_train = np.reshape(X_train, (3994, 14, 1))
X_train = np.reshape(X_train, (3994, 14))
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
shuffled_X = X_train[indices]
shuffled_y = y_train[indices]
num_classes = 5
y_train_encoded = to_categorical(shuffled_y, num_classes=num_classes)
smodel = Sequential()
smodel.add(SimpleRNN(units=64, activation='relu', input_shape=(14,1)))
smodel.add(Dense(units=num_classes, activation='softmax'))
smodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
smodel.fit(shuffled_X, y_train_encoded, batch_size=800, epochs=100)
smodel = RandomForestClassifier()
smodel.fit(shuffled_X, shuffled_y)
smodel = GradientBoostingClassifier()
smodel.fit(shuffled_X, shuffled_y)
testdf = pickle.load(open('data/df.pickle', 'rb'))
faces_t = testdf['data']
labels_t = testdf['labels']
X_test = np.array(faces_t)
y_test = np.array(labels_t)
predictions = smodel.predict(X_test)
# print(accuracy_score(y_test, predictions))

ff = open('data/FaceShapeModel.json', 'wb')
pickle.dump({'model' : smodel, 'labels': labels}, ff)
ff.close()