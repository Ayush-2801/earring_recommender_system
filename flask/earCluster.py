from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

path = "Earring_Dataset"
earrings = []

for files in os.listdir(path):
    shape_pth = os.path.join(path, files)
    if files.endswith('.png'):
        earrings.append(shape_pth)

img = load_img(earrings[0], target_size=(224,224)).convert('L')
img = np.array(img)

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    img =load_img(file, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features

data = {}
p = "data/saved.pkl"

for earring in earrings:
    try:
        feat = extract_features(earring, model)
        data[earring] = feat
    except:
        with open(p, 'wb') as file:
            pickle.dump(data, file)

filenames = np.array(list(data.keys()))
feat = np.array(list(data.values()))

feat = feat.reshape(2605, 4096)
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

kmeans = KMeans(n_clusters=10, random_state=22)
kmeans.fit(feat)

groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)
        
def view_cluster(cluster):
    plt.figure(figsize = (25,25))
    files = groups[cluster]
    num_1=1
    parent_dir = "output"
    sub=f'class{cluster}/'
    mergepath=os.mkdir(os.path.join(parent_dir,sub))
    dirpath = os.path.join(parent_dir, sub)
    for index, file in enumerate(files):
        img = load_img(file)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(dirpath+f'{str(num_1)}.png', img)
        num_1 +=1
    num_1=1
        
# print("length of group[0]",len(groups[0]))
# print("length of groups",len(groups))
for cluster in range(len(groups)):
    view_cluster(cluster)
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)
    sse.append(km.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel('Number of clusters *k*')
plt.ylabel('Sum of squared distance')