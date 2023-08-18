#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import the necessary files 

import tensorflow as tf
from tensorflow.keras import datasets, layers, models 
import matplotlib.pyplot as plt 
import numpy as np 
from keras.utils import to_categorical

# In[2]:

# load the cifar database  
(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape

# 50k training samples 
# 32 by 32 img
# 3 = rgb channels 


# In[3]:

# test the shape 
X_test.shape

# In[4]:

y_train.shape


# In[5]:

def plot_samp(X, y, index):
    plt.figure(figsize = (15, 2))
    plt.imshow(X_train[5])

# In[6]:

y_train[:5]


# In[7]:

y_train = y_train.reshape(-1,)
y_train[:5]


# In[8]:

y_test = y_test.reshape(-1,)

# In[9]:

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# In[10]:

def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

# In[11]:

plot_sample(X_train, y_train, 8)

# In[12]:

plot_sample(X_train, y_train, 4)

# In[13]:

plot_sample(X_train, y_train, 2)

# In[14]:

plot_sample(X_train, y_train, 6)

# In[15]:

X_train = X_train / 255.0
X_test = X_test / 255.0

# # Build a simple artificial neural network for image classification

# In[22]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'), #3000 neurons
        layers.Dense(1000, activation='relu'), #1000 neurons
        layers.Dense(10, activation='softmax') #10 categories of img
    ])


ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


# In[24]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# # Lets design a convolution model for training the images

# In[21]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[25]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# In[52]:

cnn.fit(X_train, y_train, epochs=10)


# # CNN > ANN as accuracy was about 70% at the end of epoch
# # Computation is less compared to ANN 

# In[51]:

cnn.evaluate(X_test,y_test)

# In[29]:

y_pred = cnn.predict(X_test)
y_pred[:5]

# In[30]:

y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]

# In[31]:

y_test[:5]

# In[43]:

plot_sample(X_test, y_test,7)

# In[42]:

classes[y_classes[7]]

# In[37]:

plot_sample(X_test, y_test,5)

# In[44]:

classes[y_classes[5]]

# In[39]:

plot_sample(X_test, y_test,3)

# In[40]:

classes[y_classes[3]]

