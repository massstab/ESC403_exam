library(keras)
library(tidyverse)
library(stringr)
library(imager)

AbdomenCT_path <- list.files(path = "med-mnist/AbdomenCT")
BreastMRI_path <- list.files(path = "med-mnist/BreastMRI")
ChestCT_path <- list.files(path = "med-mnist/ChestCT")
CXR_path <- list.files(path = "med-mnist/CXR")
Hand_path <- list.files(path = "med-mnist/Hand")
HeadCT_path <- list.files(path = "med-mnist/HeadCT")
size = 64
channels = 3
img_dataset <- c(
  AbdomenCT_path[1:length(AbdomenCT_path)], 
  BreastMRI_path[1:length(BreastMRI_path)],
  ChestCT_path[1:length(ChestCT_path)],
  CXR_path[1:length(CXR_path)],
  Hand_path[1:length(Hand_path)],
  HeadCT_path[1:length(HeadCT_path)]
)
img_dataset <- sample(img_dataset)

data_prep <- function(images, size, channels, path, list_img){
  
  count<- length(images)
  master_array <- array(NA, dim=c(count,size, size, channels))
  
  for (i in seq(length(images))) {
    folder_list <- list("AbdomenCT_path", "BreastMRI_path", "ChestCT_path", "CXR_path", "Hand_path", "HeadCT_path")
    for(j in 1:length(folder_list)) {
      if(images[i] %in% list_img[[j]]) {
        img_path <- paste0(path, folder_list[[j]], "/", images[i])
        break
      }
    }
    img <- image_load(path = img_path, target_size = c(size,size))
    img_arr <- image_to_array(img)
    img_arr <- array_reshape(img_arr, c(1, size, size, channels))
    master_array[i,,,] <- img_arr
  }
  return(master_array)
}

label_prep <- function(images, list_img) {
  y <- c()
  for(i in seq(length(images))) {
    folder_list <- list("AbdomenCT_path", "BreastMRI_path", "ChestCT_path", "CXR_path", "Hand_path", "HeadCT_path")
    for(j in 1:length(folder_list)) {
      if(images[i] %in% list_img[[j]]) {
        y <- append(y, j-1)
        break
      }
    }
  }
  return(y)
}

list_img <- list(AbdomenCT_path, BreastMRI_path, ChestCT_path, CXR_path, Hand_path, HeadCT_path)
X <- data_prep(img_dataset, size, channels, "med-mnist/", list_img)
y <- to_categorical(label_prep(img_dataset, list_img))

## Now you can proceed with splitting the dataset in train and testing