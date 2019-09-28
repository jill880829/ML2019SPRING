import sys
import csv
import numpy as np
from PIL import Image
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def read_image():
    image = []
    for i in range(40000):
        img_path = sys.argv[1]+'/%06d.jpg'%(i+1)
        img = Image.open(img_path)
        image.append(np.asarray(img))
        img.close()
    image = np.asarray(image)
    print(image.dtype)
    image = image.astype(float)
    image /= 255
    return image

img = read_image()    
data = np.genfromtxt(sys.argv[2],dtype=str,delimiter=',')
data = np.delete(data,0,0)
data = np.delete(data,0,1)
data = data.astype(int)
encoder = load_model('encoder.h5')
X = encoder.predict(img)
pca = PCA(n_components=1000,whiten=True,random_state=87)
X = pca.fit_transform(X)
clustering = KMeans(n_clusters=2,random_state=87).fit(X)

labels = clustering.labels_
res = []
idnum = []
for i,w in enumerate(data):
    idnum.append(i)
    if labels[w[0]-1] == labels[w[1]-1]:
        res.append(1)
    else:
        res.append(0)
ans = []
ans.append(idnum)
ans.append(res)
ttle = np.asarray([["id","label"]])
ans = np.asarray(ans)
ans = np.concatenate((ttle,ans.transpose()), axis = 0)
np.savetxt(sys.argv[3],ans,delimiter=',',fmt="%s")