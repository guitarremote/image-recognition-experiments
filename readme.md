I was trying to build an android app for image recognition. I hosted a server from my local machine and transferred the images captured on my mobile to my local machine by providing my server's IP in the app's backend code.

Initially I downloaded loads of images from ImageNet and thought of building my own CNN model. For that reason I wrote an R code to read the images in a folder, resize them and store the pixel data in a dataframe. Later I just used the VGG-16 CNN model and its pretrained weights. I found the tensorflow implementation of VGG-16 model online on Analytics Vidhya. 
