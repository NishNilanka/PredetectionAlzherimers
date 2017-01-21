# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:56:53 2016

@author: nishan
"""

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.utils.visualize_util import plot
#from keras.utils.drawing_utils import EpochDrawer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 160, 160

# number of channels
img_channels = 1

#  data
path1 = '/media/nishan/Entertainment/CNN/CannyExtended/'    #path of folder of images    
path2 = '/media/nishan/Entertainment/CNN/Resized_CannyExtended/'  #path of folder to save images    

listing = os.listdir(path1) 
num_samples=size(listing)
print num_samples

for file in listing:
	#Image resizing and grayscale transformation
    im = Image.open(path1 + '//' + file)   
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')           
    gray.save(path2 +'//' +  file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open('/media/nishan/Entertainment/CNN/Resized_CannyExtended/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open('/media/nishan/Entertainment/CNN/Resized_CannyExtended/' + im2)).flatten() for im2 in imlist],'f')
               
#label the data (0 - AD, 1 - MCI, 2 - NL)			   
label=np.ones((num_samples,),dtype = int)
label[0:413]=0
label[413:1213]=1
label[1213:]=2


data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]



img=immatrix[167].reshape(img_rows,img_cols)
#plt.imshow(img)
#plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)

#%%


#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 20


# number of convolutional filters to use
nb_filters = 32
# convolution kernel size
nb_conv = 3
# size of pooling area for max pooling
nb_pool = 2

#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 100
#plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

#%%

model = Sequential()

#1st Convolution Layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
#ist Activation Function						
convout1 = Activation('relu')
model.add(convout1)
#2nd Convolution Layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
#2nd Activation Function
convout2 = Activation('relu')
model.add(convout2)

#1st Pooling layer
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#Start Training Process
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
            
#print(hist.history)
print(hist.history.keys())
plt1.plot(hist.history['acc'])
plt1.plot(hist.history['val_acc'])
plt1.title('model accuracy')
plt1.ylabel('accuracy')
plt1.xlabel('epoch')
plt1.legend(['train', 'test'], loc='upper left')
plt1.show()


plt2.plot(hist.history['loss'])
plt2.plot(hist.history['val_loss'])
plt2.title('model loss')
plt2.ylabel('loss')
plt2.xlabel('epoch')
plt2.legend(['train', 'test'], loc='upper left')
plt2.show()

# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)


#%%       

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])

#plot(model, to_file='model.png')
#EpochDrawer(his, save_filename='epochs.png')

#print(model.layers[1])

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
y_pred = model.predict_classes(X_test)
print(y_pred)

p=model.predict_proba(X_test) # to predict probability

target_names = ['class 0(AD)', 'class 1(MCI)', 'class 2(NL)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))
            
fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)



# visualizing intermediate layers


output_layer = model.layers[3].output
output_fn = theano.function([model.layers[0].input], output_layer)

# the input image

input_image=X_train[0:1,:,:,:]
print(input_image.shape)

#plt.imshow(input_image[0,0,:,:],cmap ='gray')
#plt.imshow(input_image[0,0,:,:])


output_image = output_fn(input_image)
print(output_image.shape)

# Rearrange dimension so we can plot the result 
output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)
print(output_image.shape)


fig=plt.figure(figsize=(8,8))
for i in range(32):
    ax = fig.add_subplot(6, 6, i+1)
    #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    ax.imshow(output_image[0,:,:,i],cmap=matplotlib.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt


#fname = "/media/nishan/Entertainment/CNN/weights-Test-CNN.hdf5"
#model.save_weights(fname)
