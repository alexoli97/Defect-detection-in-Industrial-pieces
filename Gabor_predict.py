# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 19:35:28 2022

Segmentation prediction using RF in Gabor_train.py
STEP1: Import images that want to be predicted
STEP2: Apply the same filters and operations (than the trained one in Gabor_train.py) to these images
STEP3: Save in a folder "Spring/synthetic_gabor_result" the predicted segmentation.


@author: alexo
"""

import numpy as np
import cv2
import pandas as pd
 
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

#########################################################

#Applying trained model to segment multiple files. 

import pickle
from matplotlib import pyplot as plt

filename = "gear_model_gabor_no_drop_background"
loaded_model = pickle.load(open(filename, 'rb'))

path = "Gear/Images_15/"
import os
for image in os.listdir(path):  #iterate through each file to perform some action
    print(image)
    img1= cv2.imread(path+image)
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    #Call the feature extraction function.
    X = feature_extraction(img)
    result = loaded_model.predict(X)
    segmented = result.reshape((img.shape))
    
    #plt.imsave('Gear/Synthetic Gabor Result no drop 0/'+ image, segmented, cmap ='jet')
