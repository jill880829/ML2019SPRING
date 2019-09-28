import os
import sys
import numpy as np 
from skimage.io import imread, imsave

IMAGE_PATH = sys.argv[1]

img = imread(os.path.join(IMAGE_PATH,sys.argv[2]))
# Number of principal components used
k = 5

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

filelist = os.listdir(IMAGE_PATH) 

# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH,filelist[0])).shape 
x = 0
cnt = 0
img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH,filename))  
    if np.array_equal(tmp,img):
    	x = cnt
    cnt+=1
    img_data.append(tmp.flatten())
print(x)
training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 

# Use SVD to find the eigenvectors 
u, s, v = np.linalg.svd(np.transpose(training_data), full_matrices = False)  
# Compression
weight = np.array([s[i]*v[i,x] for i in range(k)])  

# Reconstruction
reconstruct = process(u[:,:k].dot(weight) + mean)
imsave(sys.argv[3], reconstruct.reshape(img_shape)) 

# for i in range(5):
#     number = s[i] * 100 / sum(s)
#     print(number)