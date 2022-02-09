# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 02:00:35 2022

Defect segmentation using Unet model

STEP1) Use the data_augmentation.py folder to create more training images. I created a folder with 2000 and 600 (I can't use the 2000 one due to memory allocation problems')
STEP2)Read the images and masks (Change Gear with Spring in the folder name for choosing one or other), resize them and save them into a np.array
STEP3) Normalize and expend dimensions to be able to use Unet Model
STEP4) Split data intro train/test
STEP5) Define a Unet Model (In my case I define it in other script called simple_unet_model)
STEP6)Train the model and save it
STEP7) Check accuracy and IoU (better measurement for image segmentation)
STEP8) Test the model in a random test image

@author: alexo
"""
import tensorflow as tf
from simple_unet_model import simple_unet_model   
from  tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
##
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

###################################
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Activate GPU in 0 , deactivate in -1


image_directory = 'Gear/Augmented Synthetic Images 600/'
mask_directory = 'Gear/Augmented Synthetic Masks 600/'


SIZE = 256
image_dataset = []  #Using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 256 x 256
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))


#Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
#Not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

###################################
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]
###################################
#Model 1
def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()


history = model.fit(X_train, y_train, 
                    batch_size = 3,  #low batch_size due to small memory size 
                    verbose=1, 
                    epochs=30, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save('gear_model_Unet.hdf5')

##################################
#Evaluate the model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##################################
#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5 #Use threshold for better measurement of prediction

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is: ", iou_score)

##################################
#Predict on a test image
#model = get_model()
#model.load_weights('gear_model_Unet.hdf5')  

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.3).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)

##################################
#Predict on created synthetic image

test_img_other2 = cv2.imread('Gear/Blender_Synthetic_Images/1 (4).png', 0) #Select the path of the desired image to test
test_img_other2 = cv2.resize(test_img_other2, (SIZE, SIZE))
test_img_other_norm2 = np.expand_dims(normalize(np.array(test_img_other2), axis=1),2)
test_img_other_norm2=test_img_other_norm2[:,:,0][:,:,None]
test_img_other_input2=np.expand_dims(test_img_other_norm2, 0)

prediction_other2 = (model.predict(test_img_other_input2)[0,:,:,0] > 0.3).astype(np.uint8)

plt.title('Created synthetic image')
plt.imshow(test_img_other2, cmap='gray')
plt.title('Prediction on test image')
plt.imshow(prediction_other2, cmap='gray')
