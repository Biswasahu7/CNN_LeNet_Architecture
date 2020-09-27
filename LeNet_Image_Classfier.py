#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential


# In[29]:


model = Sequential()


# In[30]:


#LeNet_CNN_Architecture
model.add(Conv2D(6, kernel_size= (5, 5), activation = 'relu', input_shape= (28,28,3)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(120, activation= 'relu'))
model.add(Dense(84, activation = 'relu'))
model.add(Dense(3, activation='relu'))


# In[31]:


#model.summary()


# In[34]:


model.compile(loss=keras.metrics.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('G:\FaMily',
                                                 target_size = (28, 28),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('G:\FaMily',
                                            target_size = (28, 28),
                                            batch_size = 32,
                                            class_mode = 'binary')


model.fit_generator(training_set,
                         steps_per_epoch = 1500,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 50)


# In[ ]:




