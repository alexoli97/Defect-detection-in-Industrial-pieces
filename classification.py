# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 00:52:57 2022

Classify objects in Gear or Spring and apply the corresponding defect detection models to an image
STEP1: Train a model using CNN 
STEP2: Import an image and predict if it is a Gear or Spring
    For that, an input folder called "Input_Images" is used, where the first 2 images are Synthetic gear, 
    the 2 next are Real gear, the 2 next are Synthetic Spring and the last 2 are real Spring.
STEP3: Apply the correspoding pretrained Unet model to the image
STEP4: Apply the corresponding pretained RF (using filters) model to it
STEP5: In the output folder we can see comparison between models. 

@author: alexo
"""

import numpy as np 
from  tensorflow.keras.utils import normalize
import matplotlib.pyplot as plt
import glob
import cv2
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import os
import seaborn as sns
from simple_unet_model import simple_unet_model 

print(os.listdir("Classification/"))

SIZE = 128

images = []
labels = [] 
for directory_path in glob.glob("Classification/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
        labels.append(label)
        
images = np.array(images)
labels = np.array(labels)


# Split train-test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.10, random_state = 0)
 
x_train = np.array(x_train) 
y_train = np.array(y_train)      
x_test = np.array(x_test)
y_test = np.array(y_test)

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_test)
y_test_encoded = le.transform(y_test)
le.fit(y_train)
y_train_encoded = le.transform(y_train)


###################################################################
# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network. 
from tensorflow.keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

#############################

activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (SIZE, SIZE, 3)))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())

#Add layers for deep learning prediction
x = feature_extractor.output  
x = Dense(128, activation = activation, kernel_initializer = 'he_uniform')(x)
prediction_layer = Dense(2, activation = 'softmax')(x)

# Make a new model combining both feature extractor and x
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(cnn_model.summary()) 

##########################################
val_acc=[]
while True:
    
    history = cnn_model.fit(x_train, y_train_one_hot, epochs=25, validation_data = (x_test, y_test_one_hot))
    

    #Save the model 
    cnn_model.save('Classification_model.hdf5')

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
    if (val_acc[24]>0.90):  #Sometimes the model gets a lot of change between epochs for some reason I don't know. If a bad result, repeat. 
        break
    
#Train the CNN model
prediction_NN = cnn_model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True)

#################################
#Predict on input image
######################################

test_img = cv2.imread('Input_Images/1 (5).png', cv2.IMREAD_COLOR) #Select the path of the desired image to test
test_img = cv2.resize(test_img, (128, 128)) #Match size of classification model
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = test_img/255 #normalize like input
plt.imshow(test_img)
input_img = np.expand_dims(test_img, axis=0) #Expand dims so the input is (num images, x, y, c)
print("The probabilities are (Gear/Spring): ", cnn_model.predict(input_img)) 
prediction = cnn_model.predict(input_img)
prediction = np.argmax(cnn_model.predict(input_img))  #argmax to convert categorical back to original
prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction)

######################################
#Apply defect model depending on which image is predicted
######################################

IMG_HEIGHT = 256 #to match the size of the pretained weights
IMG_WIDTH  = 256
IMG_CHANNELS = 1

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()
if prediction=='Gear' :
   model.load_weights('gear_model_Unet.hdf5') 
   print("gear_model_Unet selected")
elif prediction=='Spring' :
   model.load_weights('spring_model_Unet.hdf5')
   print("spring_model_Unetselected")

test_img_other = cv2.imread('Input_Images/1 (5).png', 0) #Select the path of the desired image to test again (should be the same as above)
test_img_other = cv2.resize(test_img_other, (256, 256)) #Match size of pretained weights
test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
test_img_other_input=np.expand_dims(test_img_other_norm, 0)

#Predict and threshold for values above 0.5 probability
prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.5).astype(np.uint8)
#save image
plt.imsave('Prediction_Input_Image/1 (5).png', prediction_other, cmap='gray') #Select the path to save the output image

######################################
#Do the same but with the Traditional model using filters to compare 
######################################
import pickle
from matplotlib import pyplot as plt
import pandas as pd

if prediction=='Gear' :
    filename = "gear_model_gabor"
    print("Gear_model_gabor selected")
elif prediction=='Spring':
    filename = "spring_model_gabor"
    print("Spring_model_gabor selected")
    
loaded_model = pickle.load(open(filename, 'rb'))

test_img_other2 = cv2.imread('Input_Images/1 (8).png', cv2.IMREAD_COLOR)
print(test_img_other2 )
test_img_other2 = cv2.cvtColor(test_img_other2,cv2.COLOR_BGR2GRAY)
    
def feature_extraction(img):
    df = pd.DataFrame()

#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Original Image'] = img2

#Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):   #3 thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 4):  #Sigma with 1 and 4
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                    #print(gabor_label)  #sanity check
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
                    
        
########################################
#Gerate OTHER FEATURES and add them to the data frame
                
    #CANNY EDGE
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    

    from scipy import ndimage as nd

    #GAUSSIAN with sigma=3
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img3 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img3
    #GAUSSIAN with sigma=5
    gaussian_img = nd.gaussian_filter(img, sigma=5)
    gaussian_img5 = gaussian_img.reshape(-1)
    df['Gaussian s5'] = gaussian_img5
    #GAUSSIAN with sigma=7
    gaussian_img = nd.gaussian_filter(img, sigma=7)
    gaussian_img7 = gaussian_img.reshape(-1)
    df['Gaussian s7'] = gaussian_img7
    
    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img3 = median_img.reshape(-1)
    df['Median s3'] = median_img3
    
    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img3 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img3  #Add column to original dataframe

    return df
#Call the feature extraction function.
X = feature_extraction(test_img_other2)
result = loaded_model.predict(X)
segmented = result.reshape((test_img_other2.shape))
    
plt.imsave('Prediction_Input_Image/Gabor 1(8).png', segmented, cmap ='gray')



