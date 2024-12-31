import cv2
import pickle
import numpy as np
import os
from PIL import Image
import random
import dlib
from imutils import face_utils
import math

class FaceShapeDetector:
    def __init__(self, model_path, shape_predictor_path, shapemodel_path):
        self.earringscollection = []
        self.foldersname = []
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.shapemodel = pickle.load(open(shapemodel_path, 'rb'))['model']
        self.model_path = model_path

    def distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def angle(self, p1, p2, p3):
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
        return math.degrees(angle_rad)

    def midpoint(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def detect_face_shape(self, image_path):
        frame = cv2.imread(image_path)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.detector(img)
        for face in dets:
            shape = self.shape_predictor(img, face)
            coords = face_utils.shape_to_np(shape)
            for (x, y) in coords:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            l3, l15, l70, l76, l80, l73 = coords[3], coords[15], coords[70], coords[76], coords[80], coords[73]
            l9, l13, l5, l7, l11, l8, l10 = coords[9], coords[13], coords[5], coords[7], coords[11], coords[8], coords[10]
            d1 = self.distance(l3, l15)
            d2 = self.distance(l76, l80)
            d3 = self.distance(self.midpoint(l70, l73), l9)
            d4 = self.distance(l9, l13)
            d5 = self.distance(l5, l13)
            d6 = self.distance(l7, l11)
            d7 = self.distance(l8, l10)
            DD = d1 + d2 + d3 + d4 + d5 + d6 + d7
            D1, D2, D3, D4, D5, D6, D7 = d1/DD, d2/DD, d3/DD, d4/DD, d5/DD, d6/DD, d7/DD
            R1, R2, R3, R4, R5, R6, R7, R8, R9, R10 = D2/D1, D1/D3, D2/D3, D1/D5, D6/D5, D4/D6, D6/D1, D5/D2, D4/D5, D7/D6
            A1 = self.angle(self.midpoint(l70, l73), l9, l11)
            A2 = self.angle(self.midpoint(l70, l73), l9, l13)
            A3 = self.angle(l3, l15, l13)
            features = np.array([R1, R2, R3, R4, R7, R8, R10, D1, D2, D3, D5, D6, A1, A2]).reshape(1, -1)
            predshape = self.shapemodel.predict(features)
            face_shape = ["Heart", "Oblong", "Oval", "Round", "Square"][predshape[0]]
            cv2.putText(frame, "Face Shape:" + face_shape, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # cv2.imshow("Detected Face Shape", frame)
            return face_shape

    def get_random_image_path(self, folder_path):
        head_tail = os.path.split(folder_path)
        self.foldersname.append(head_tail[1])
        try:
            files = os.listdir(folder_path)
            images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                # print("No images found in the specified folder.")
                return None

            random_image = random.choice(images)
            # print("random image path")
            return os.path.join(folder_path, random_image)

        except Exception as e:
            print(f"An error occurred while selecting image randomly: {e}")
            return None

    def display_image(self, image_path):
        try:
            if image_path and os.path.isfile(image_path):
                with Image.open(image_path) as img:
                    # img.show()
                    # print(f"Displayed image: {image_path}")
                    self.earringscollection.append(image_path)
            else:
                print(f"Invalid image path: {image_path}")
        except Exception as e:
            print(f"An error occurred while displaying the image: {e}")
        return self.earringscollection

    def show_random_image_from_folder(self, folder_path):
        random_image_path = self.get_random_image_path(folder_path)
        output=self.display_image(random_image_path)
        return output 
    
    def recommend_earrings(self, face_shape):
        folder_paths = {
            "heart" : ['final/desert'],
            "oblong": ['final/dangle', 'final/hoops'],
            "oval"  : ['final/chandeliers', 'final/hoops', 'final/studs', 'final/triangle', 'final/teardrop', 'final/jhumkas'],
            "round" : ['final/dangle', 'final/chandbalis'],
            "square": ['final/hoops', 'final/studs']
        }
        face_shape = face_shape.lower()
        if face_shape in folder_paths:
            paths = folder_paths[face_shape]
            if isinstance(paths, list):
                for path in paths:
                    if os.path.isdir(path):
                        earrings = (self.show_random_image_from_folder(path), self.foldersname)
                    else:
                        print(f"The specified folder does not exist: {path}")
            elif os.path.isdir(paths):
                earrings = (self.show_random_image_from_folder(paths), self.foldersname)
            else:
                print(f"The specified folder does not exist: {paths}")
        return earrings
        # return self.show_random_image_from_folder(path)