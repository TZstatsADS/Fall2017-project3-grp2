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
hog_features = read_csv("./doc/hog.csv")

label = read_csv("~/Downloads/training_set/label_train.csv")

data = data.frame(label[,2], sift_train[,2:ncol(sift_train)])
data_hog = data.frame(label[,2], hog_features[,2:ncol(hog_features)])

colnames(data)[1] = "label"
colnames(data_hog)[1] = "label"

#######################################
# divide into train & test for SIFT (70:30)
#######################################
set.seed(123)
index = sample(1:nrow(data), size=0.7*nrow(data))
train_data = data[index,]
test_data = data[-index,]

#######################################
# divide into train & test for HOG (70:30)
#######################################
set.seed(123)
index = sample(1:nrow(data_hog), size=0.7*nrow(data_hog))
train_data = data_hog[index,]
test_data = data_hog[-index,]

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

#system.time(multinom_train(train_data))

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
# For SIFT
#Accuracy     Kappa 
#0.7254464 0.5882907 
#-> error rate 27.45%

# For HOG
#Accuracy     Kappa 
#0.8044444 0.7065221 
#-> error rate 19.56%


#confusionMatrix(test_data$label,multinomtest_result)

