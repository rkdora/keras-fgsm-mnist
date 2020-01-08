#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import RMSprop

from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

import time


# In[2]:


start = time.time()


# In[3]:


def generate_p(x, label, model):
    class_output = model.output[:, int(label)]
    
    grads = K.gradients(class_output, model.input)[0]
    gradient_function = K.function([model.input], [grads])

    grads_val = gradient_function([x.reshape(1, 784)])
    
    p = np.sign(grads_val)
    
    return p


# In[4]:


def generate_adv(x, label, model, eps):
    p = generate_p(x, label, model)
    adv = (x - eps*p).clip(min=0, max=1).reshape(1, 784)
    
    return adv


# In[5]:


def predict(x, model):
    pred = model.predict(x.reshape(1,784), batch_size=1)
    pred_class = np.argmax(pred)
    pred_per = max(pred[0])
    
    return pred_class, pred_per


# In[6]:


eps = 0.3


# In[7]:


num_classes = 10


# In[8]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test  = X_test.reshape(10000, 784).astype('float32') / 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train_catego = keras.utils.to_categorical(y_train, num_classes)
y_test_catego = keras.utils.to_categorical(y_test, num_classes)


# In[9]:


# モデルを読み込む
model = model_from_json(open('mnist_mlp_model.json').read())

# 学習結果を読み込む
model.load_weights('mnist_mlp_weights.h5')

model.summary();

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# In[10]:


score = model.evaluate(X_test, y_test_catego, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])


# In[11]:


idx = 1
x = X_test[idx]
y = y_test[idx]
plt.imshow(x.reshape(28,28), 'gray')
plt.show()


# In[12]:


pred_class, pred_per = predict(x, model)
print(pred_class, pred_per)


# In[13]:


adv = generate_adv(x, y, model, eps)
plt.imshow(adv.reshape(28,28), 'gray')
plt.show()


# In[14]:


pred_class, pred_per = predict(adv, model)
print(pred_class, pred_per)


# In[15]:


def generate_adv_list(x_list, y_list, model, eps):
    adv_list = []
    
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        adv = generate_adv(x, y, model, eps).reshape(784)
        adv_list.append(adv)
        print(i)
        
    return np.array(adv_list)


# In[16]:


max_n = 1000


# In[17]:


adv_test = generate_adv_list(X_test[:max_n], y_test[:max_n], model, eps)
adv_test.shape


# In[18]:


score = model.evaluate(adv_test, y_test_catego[:max_n], verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])


# In[19]:


print(time.time() - start)


# In[ ]:




