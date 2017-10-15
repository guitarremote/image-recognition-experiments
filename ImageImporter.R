#Load Libraries

library(imager)
library(EBImage)


#Remove all existing variables
rm(list=ls())

#Set up word directory
setwd("C:/Users/Aravind Atreya/Desktop/Kaggle/Image Data/Pics")
products <- list.files()
total_df <- rep(0,784)

for (i in 1:length(products)){
  files <- c()
  files <- c(files,list.files(products[i]))
  # create progress bar
  pb <- txtProgressBar(min = 0, max = length(files), style = 3)
  for(j in 1: length(files)){
    
    #update progress bar
    setTxtProgressBar(pb, j)
    paths <-c()
    paths<-paste0(products[i],"/",files[j])
    img <- readImage(paths)
    img <- channel(img,"gray")
    img <- resize(img,w=28,h=28)
    df <- as.vector(t(img))
    total_df <- rbind(total_df,df)
  }
  close(pb)
}

total_df <- total_df[-1,]
dim(total_df)

