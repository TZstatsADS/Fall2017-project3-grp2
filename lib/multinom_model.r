rm(list=ls())
setwd('~/Documents/Github/Fall2017-project3-fall2017-project3-grp2')

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
#library(gbm)

#######################################
# Load sift features + labels
#######################################
sift_train = read_csv("~/Downloads/training_set/sift_train.csv")
label = read_csv("~/Downloads/training_set/label_train.csv")
data = data.frame(label[,2], sift_train[,2:ncol(sift_train)])
colnames(data)[1] = "label"

#######################################
# divide into train & test (70:30)
#######################################
set.seed(123)
index = sample(1:nrow(data), size=0.7*nrow(data))
train_data = data[index,]
test_data = data[-index,]

#######################################
# multinom_train function
#######################################
multinom_train <- function(train_data){
  multinom_fit <- multinom(formula = as.factor(label) ~ .,
                           data=train_data, MaxNWts = 100000, maxit = 500)
  top_models = varImp(multinom_fit)
  top_models$variables = row.names(top_models)
  top_models = top_models[order(-top_models$Overall),]
  return(list(fit=multinom_fit, top=top_models))
}

# run it:
multinomfit_train = multinom_train(train_data)

#######################################
# multinom_test function
#######################################
multinom_test <- function(test_data, fit){
  multinom_pred = predict(fit, type="class", newdata=test_data)
  return(multinom_pred)
}

#run it:
multinomtest_result = multinom_test(test_data,multinomfit_train$fit)
postResample(test_data$label,multinomtest_result)

#> postResample(test_data$label,multinomtest_result)
#Accuracy     Kappa 
#0.7254464 0.5882907 

#confusionMatrix(test_data$label,multinomtest_result)







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
