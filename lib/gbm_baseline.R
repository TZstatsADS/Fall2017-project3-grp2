#######################################
# install & load necessary packages
#######################################
packages.used=c("readr","nnet", "caret","e1071","gbm")

packages.needed=setdiff(packages.used, intersect(installed.packages()[,1], packages.used))
if(length(packages.needed)>0){
  install.packages(packages.needed, dependencies = TRUE,repos='http://cran.us.r-project.org')
}
library(readr)
library(nnet)
library(caret)
library(gbm)

#######################################
# Load sift features + labels
#######################################
setwd("C:/Users/enriquethemoist/Dropbox/Columbia Work/4A - one/Applied Data Science/Project 3/training set")
sift_train = read_csv("sift_train.csv")
label = read_csv("label_train.csv")
data = data.frame(label[,2], sift_train[,2:ncol(sift_train)])
colnames(data)[1] = "label"

#######################################
# base model example: multinomual version
#######################################
gbm_train <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  
  ### Train with gradient boosting model
  if(is.null(par)){
    depth <- 3
  } else {
    depth <- par$depth
  }
  fit_gbm <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=2000,
                     distribution='multinomial',
                     interaction.depth=depth, 
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
  
  return(list(fit=fit_gbm, iter=best_iter))
}

gbmfit_train = gbm_train(train_data[,2:ncol(train_data)],train_data$label)
tm_gbmfit_train <- system.time(gbm_train(train_data[,2:ncol(train_data)],train_data$label))

gbm_test <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  pred <- predict(fit_train$fit, newdata=dat_test, 
                  n.trees=fit_train$iter, type="response")
  
  return(as.numeric(pred> 0.5))
}

gbmtest_result = gbm_test(fit_train, test_data)
postResample(test_data$label,gbmtest_result)