#!/usr/bin/env python
# coding: utf-8

# In[495]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from collections import Counter

import numpy as np



from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
#from keras.layers.normalization import layer_normalization
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import csv



os.chdir(r"D:\ajmera sop\MSpalmprint ROI Database\ROI Database")


# In[496]:


counter=0


# In[497]:


strs = ["" for x in range(500)]
for i in range(5):
    for j in range(10):
        for k in range(10):
            strs[i*100+j*10+k]=str(0)+str(i)+str(j)+str(k)
            
for i in range(500):
    if(i<499):
        strs[i]=strs[i+1]
    else :
        strs[499]=str(0)+str(500)
        
strs


# In[498]:


l_r = [0] * 6000
label = [0]*6000
def convert(img):
    return np.array(img)
for i in range (500):
    os.chdir(r"D:\ajmera sop\MSpalmprint ROI Database\ROI Database\Red\{folder}".format(folder=strs[i]))
    for j in range (12):
        all_files = os.listdir()
        img= Image.open(all_files[j])
#         df['images'] = convert(img)
        l_r[i*12+j]=convert(img)
        counter+=1
        label[i*12+j]=i+1
        
            
        


# In[499]:


l_b = [0] * 6000
label = [0]*6000
def convert(img):
    return np.array(img)
for i in range (500):
    os.chdir(r"D:\ajmera sop\MSpalmprint ROI Database\ROI Database\Blue\{folder}".format(folder=strs[i]))
    for j in range (12):
        all_files = os.listdir()
        img= Image.open(all_files[j])
#         df['images'] = convert(img)
        l_b[i*12+j]=convert(img)
        counter+=1
        label[i*12+j]=i+1
        


# In[500]:


l_g = [0] * 6000
label = [0]*6000
def convert(img):
    return np.array(img)
for i in range (500):
    os.chdir(r"D:\ajmera sop\MSpalmprint ROI Database\ROI Database\Green\{folder}".format(folder=strs[i]))
    for j in range (12):
        all_files = os.listdir()
        img= Image.open(all_files[j])
#         df['images'] = convert(img)
        l_g[i*12+j]=convert(img)
        counter+=1
        label[i*12+j]=i+1
        


# In[501]:


l_n = [0] * 6000
label = [0]*6000
def convert(img):
    return np.array(img)
for i in range (500):
    os.chdir(r"D:\ajmera sop\MSpalmprint ROI Database\ROI Database\NIR\{folder}".format(folder=strs[i]))
    for j in range (12):
        all_files = os.listdir()
        img= Image.open(all_files[j])
#         df['images'] = convert(img)
        l_n[i*12+j]=convert(img)
        counter+=1
        label[i*12+j]=i+1
        


# In[502]:


r=np.array(l_r)
g=np.array(l_g)
b=np.array(l_b)
n=np.array(l_b)
final=np.array((l_r,l_g,l_b,l_n))


# In[503]:


#se=pd.Series(l_r)
se=pd.Series(final)
print(se)


# In[504]:


la=pd.Series(label)


# In[505]:


det = pd.concat([se,la], join = 'outer', axis = 1)


# In[506]:


det.rename(columns = {'0' : 'Image', '1' : 'Label'}, inplace = True)


# In[507]:


df.to_csv(r'D:\ajmera sop\file1.csv')


# In[508]:


dataf


# In[509]:


n=500
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(input_shape=(128,128,4),filters=16,kernel_size=(5,5),padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(1,1),padding="same"))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(11,11),padding="same"))
model.add(tf.keras.layers.LayerNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2)))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(5,5), padding="same"))
model.add(tf.keras.layers.LayerNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2)))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096,activation="relu"))
model.add(tf.keras.layers.Dense(n))


# In[510]:


model.summary()


# In[511]:


opt = tf.keras.optimizers.SGD(learning_rate=0.00001, decay=1e-6)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])


# In[512]:


print(final.shape)
result = final[:, 0,:,:]
result_f=np.swapaxes(final,1,3)
final_result=np.transpose(result)
print(result_f.shape)
result_final=np.swapaxes(result_f,0,3)
print(result_final.shape)
print(la.shape)


# In[513]:


la=np.array(la)
print(la.shape)


# In[514]:


dataf=pd.read_csv(r"D:\ajmera sop\file1.csv")
(x_train, y_train)= result_final,la

# Preprocess the data (these are NumPy arrays)
# x_train = x_train.reshape(60000, 784).astype("float32") / 255
# x_test = x_test.reshape(10000, 784).astype("float32") / 255
history = model.fit(result_final,la,epochs = 20 ), #validation_data = (x_val, y_val))



# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(20)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

