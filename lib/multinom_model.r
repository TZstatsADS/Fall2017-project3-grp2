#######################################
# Multinomial logictic regression
# Author: Saaya Yasuda (sy2569)
#######################################

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

#######################################
# Load sift features + labels
#######################################

# feature files
sift_train = read_csv("~/Downloads/training_set/sift_train.csv")
hog_features = read_csv("./doc/hog.csv")

# label
label = read_csv("~/Downloads/training_set/label_train.csv")

# merging them
data = data.frame(label[,2], sift_train[,2:ncol(sift_train)])
data_hog = data.frame(label[,2], hog_features[,2:ncol(hog_features)])

colnames(data)[1] = "label"
colnames(data_hog)[1] = "label"

#######################################
# For SIFT: divide into train & test (70:30)
#######################################
set.seed(123)
index = sample(1:nrow(data), size=0.7*nrow(data))
train_data = data[index,]
test_data = data[-index,]

#######################################
# For HOG: divide into train & test (70:30)
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

stepwisefit = step(multinomfit_train$fit, direction="both", 
                   scope=formula(multinomfit_train$fit))
#system.time(step(multinomfit_train$fit, direction="both", 
#                 scope=formula(multinomfit_train$fit)))

#######################################
# Running time
#######################################

# For SIFT
#user  system elapsed 
#360.566   2.307 365.568

# For HOG
#user  system elapsed 
#0.690   0.015   0.706 

# For HOG (via Stepwise function to reduce # of var in x)
#user  system elapsed 
#292.679   6.325 298.518 

#######################################
# multinom_test function
#######################################
multinom_test <- function(test_data, fit){
  multinom_pred = predict(fit, type="class", newdata=test_data)
  return(multinom_pred)
}

# run it:
multinomtest_result = multinom_test(test_data,multinomfit_train$fit)
postResample(test_data$label,multinomtest_result)

# only for stepwise
multinomtest_result_stepwise = multinom_test(test_data,stepwisefit)
postResample(test_data$label,multinomtest_result_stepwise)

#confusionMatrix(test_data$label,multinomtest_result)

#######################################
# Error rate / Accuracy
#######################################

# For SIFT
# Accuracy     Kappa 
# 0.7254464 0.5882907 
# -> error rate 27.45%

# For HOG
# Accuracy     Kappa 
# 0.8044444 0.7065221 
# -> error rate 19.56%

# For HOG (via Stepwise function to reduce # of var in x)
# Accuracy     Kappa 
# 0.8033333 0.7048437 
# -> error rate 19.67%


