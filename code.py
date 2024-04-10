#!/usr/bin/env python
# coding: utf-8

# # Data Pre-processing

# In[1]:


import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# - Importing libraries
# - classifying (no_tumor attribute to 0 and pituitary_tumor to 1 )

# In[3]:


path = os.listdir(r"C:/Users/Asus/Documents/brain_tumor/Training")
classes= {'no_tumor':0, 'pituitary_tumor':1 }


# 1.This line constructs the path to the directory containing images for the current class. 
# For example, when cls is 'no_tumor', pth will be "C:/Users/Asus/Documents/brain_tumor/Training/no_tumor". 
# Similarly, when cls is 'pituitary_tumor', pth will be "C:/Users/Asus/Documents/brain_tumor/Training/pituitary_tumor"
# 
# 2.This inner loop iterates over all files (images) in the directory specified by pth. For each image file (j) in the directory, it performs the following operations
# 
# 3.This line reads the image file using OpenCV's imread() function. It constructs the full path to the image by concatenating pth (the directory path) with the filename (j). The second argument 0 indicates that the image should be read in grayscale mode.
# 
# 4.This line resizes the image to a fixed size of 200x200 pixels using OpenCV's resize() function.
# 
# -img = cv2.imread(pth+'/'+j, 0): This line reads the image file named j in grayscale (0 parameter) from the path formed by   concatenating pth with j. cv2.imread() is a function from the OpenCV library used to read an image from a file.
# 
# -img = cv2.resize(img, (200,200)): This line resizes the read image to a fixed size of 200x200 pixels. cv2.resize() is a     function from the OpenCV library used to resize images.

# In[5]:


x = []
y = []

for cls in classes:
    pth = "C:/Users/Asus/Documents/brain_tumor/Training/"+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        x.append(img)
        y.append(classes[cls])


# In[6]:


np.unique(y)


# In[10]:


X=np.array(x)
Y=np.array(y)


# In[11]:


pd.Series(Y).value_counts()


# In[13]:


X.shape


# # Visualisation

# In[19]:


plt.imshow(X[0], cmap='Accent_r')


# In[21]:


X_updated = X.reshape(len(X), -1)
X_updated.shape


# In[22]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size= .20)


# In[23]:


xtrain.shape, xtest.shape


# # Feature Scaling

# In[24]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtrain.min())

xtrain = xtrain/255
xtest = xtest/255

print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# # Feature Selection: PCA

# In[25]:


from sklearn.decomposition import PCA


# In[30]:


print(xtrain.shape, xtest.shape)
pca = PCA(.98)
pca_train = xtrain
pca_test = xtest


# # Training Model

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


# In[31]:


lg = LogisticRegression(C=0.1)
lg.fit(pca_train, ytrain)


# In[32]:


sv = SVC()
sv.fit(pca_train, ytrain)


# # Evaluation

# In[36]:


print("Training Score:", lg.score(pca_train, ytrain))
print("Testing score:", lg.score(pca_test, ytest))


# In[37]:


print("Training Score:", sv.score(pca_train, ytrain))
print("Testing Score:", sv.score(pca_test, ytest))


# # Prediction

# In[38]:


pred = sv.predict(pca_test)
np.where(ytest!=pred)


# In[39]:


pred[6]


# In[40]:


ytest[6]


# # Test Model

# In[41]:


dec = {0:"No Tumor", 1:"Positive"}


# In[42]:


plt.figure(figsize=(12,8))
p = os.listdir(r"C:/Users/Asus/Documents/brain_tumor/Testing")
c = 1
for i in os.listdir("C:/Users/Asus/Documents/brain_tumor/Testing/no_tumor/")[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread("C:/Users/Asus/Documents/brain_tumor/Testing/no_tumor/"+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='Accent_r')
    plt.axis("off")
    c+=1


# In[43]:


plt.figure(figsize=(12,8))
p = os.listdir(r"C:/Users/Asus/Documents/brain_tumor/Testing")
c = 1
for i in os.listdir("C:/Users/Asus/Documents/brain_tumor/Testing/pituitary_tumor/")[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread("C:/Users/Asus/Documents/brain_tumor/Testing/pituitary_tumor/"+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='Accent_r')
    plt.axis("off")
    c+=1


# In[ ]:




