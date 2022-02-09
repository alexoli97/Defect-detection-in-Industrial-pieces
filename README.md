# Fraunhofer 

# Programmed solution and documents the obtained results

The first part about how I did this task, is that I created a new folder called "InspectionTry" which I attached in the mail. This zip file has all the python scripts (which are also in the GitHub page), but also there are some folders with the images and masks. This "InspectionTry" has the following files:

  -Gear (With all files about the Gear) -> Inside it, there are real images,synthetic images, masks and some other folders which I created.
  It is worth mentioning the folder "labels" inside Gear, which was created for the purpose of performing the last part of the task. It contains 78 gear images and it corresponding masks but with annotations on it. that is, I used an annotation tool on "Apeer.com" website in which I manually annotated 2 types of defects: Pitting (round shaped defects) and Scratching (long thin defects with curves). The result of these annotations are masks in .tiff format with annotation, that is, the background has value 0, the pitting defect has value 1 and the scratching defect has value 2.   
  
  -Spring (With all files about the Spring ) -> Inside it, there are real images,synthetic images, masks and some other folders which I created.
  It is worth mentioning the folder Blender_Synthetic_Images ,in both Gear and Spring, which contains some images taken by me in the software Blender.
  -Classification (labels for defect classification) -> Inside it, there are two subfolders, one with all Gear images (synthetic and real) and the other with all Spring images (synthetic and real)
  
  -Input Images -> A selection of 8 test images (2 synthetic gear, 2 real gear, 2 synthetic spring, 2 real spring) used for testing the results of the models in the different types of images.
  
  -Prediction Input Image -> The output of the models that use "Input Images"
  
  -Most of the scripts that are in the GitHub page (except two: DefectSegmentationUnet.py and Defect_classification.ipynb)
  -Some pretrained weights for the models.
 
The scripts in Github are:

- DefectSegmentationUnet.py -> model for defect segmentation using Unet CNN. Good results are obtained in the gear, especially in the synthetic images. On the other hand, poor results are obtained when training the springs because of the greater complexity of the images with respect to the gear. in the final part of this code an image taken in the blender software ("blender_synthetic_images" subfolder) is included to test the model. Good results are obtained for the gear and bad results for the spring, as expected. This script uses other two scripts:
  - simple_unet_model.py -> It defines the architecture of the Unet model.
  - data.augmentation.py -> Code used for increasing the size of the dataset. We only had around 120 images for gear, that might not be enough for a Unet CNN model. Therefore , use this code for increasing Gear and Spring images to 600 (Not more due to memory allocation problems in my GPU)

-Gabor.train.py and Gabor.predict.py -> model for defect segmentation using Random Forest and image filters. Due to the not so satisfactory results with the spring, I decided to try another type of model, in this case using several image filters, mainly gabor filters and creating a model with random forest. Better accuracy results are obtained although equal or worse visualization of the defects in the output image. It is worth mentioning, that the input is only 15 images (subfolders images_15 and masks_15 in both gear and spring) since this is not a Deep learning model that requires large amounts of input data. The output of this code is shown in the folder: Gabor result in both gear and spring.

-classification.py -> Classification model which can recognize if the image contains spring or gear object. After, a image is selected from the folder "Input Images" and this program recognizes which object it is and runs appropriate segmentation model. It does the segmentation in both models described previosly, that is, Unet and RF.

-Defect_classification.py -> A simple model using CNN (extractor) + RF to try to predict what kind of defect an image has. Bad results are obtained.

-Defect_classification.ipynb -> Since I was not able to obtain good results for the defect classification. I tried more complex deep learning models using different BACKBONES (resnet34,inceptionv3 and vgg16). However, the model only predicts well the background and not the defect. even adding different weights in the models to give more value to the defect pixels. 

In summary, the task was acceptably accomplished except for the prediction of springs and classification of defect types.

For this task I used spyder(anaconda tensorflow) and Google colab for one of the scripts (Defect_classification.ipynb)


  
  
