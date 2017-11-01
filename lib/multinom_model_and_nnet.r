#######################################
# Multinomial logictic regression & Neural Net
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
hog_features = read_csv("./data/hog.csv")
lbp_features = read_csv("./data/lbp.csv",col_names=FALSE)

# label
label = read_csv("~/Downloads/training_set/label_train.csv")

# merging them
data = data.frame(label[,2], sift_train[,2:ncol(sift_train)])
data_hog = data.frame(label[,2], hog_features[,2:ncol(hog_features)])
data_lbp = data.frame(label[,2], lbp_features[,1:ncol(lbp_features)])

colnames(data)[1] = "label"
colnames(data_hog)[1] = "label"
colnames(data_lbp)[1] = "label"

source("../lib/eco2121_train_gbm_baseline.r")

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
# For LBP: divide into train & test (70:30)
#######################################
set.seed(123)
index = sample(1:nrow(data_lbp), size=0.7*nrow(data_lbp))
train_data = data_lbp[index,]
test_data = data_lbp[-index,]

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

# For LBP
#user  system elapsed 
#1.294   0.027   1.335 

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

# For LBP
# Accuracy     Kappa 
# 0.7633333 0.6450079 


#######################################
# nnet_train function
#######################################
nnet_train <- function(train_data, size){
  nnet_fit <- nnet(formula = as.factor(label) ~ .,
                   data=train_data, MaxNWts = 100000, 
                   maxit = 2000, size = size, trace=FALSE) #maxit is set based on a few tests
  return(nnet_fit)
}

#######################################
# nnet_test function
#######################################

nnet_test <- function(test_data, fit){
  nnet_pred = predict(fit, type="class", newdata=test_data)
  return(nnet_pred)
}

cv <- function(train_data,test_data){
  accuracy_vec = c()
  for(i in 1:5){
    nnetfit_train = nnet_train(train_data, i)
    nnettest_result = nnet_test(test_data,nnetfit_train)
    accuracy = postResample(as.factor(test_data$label),as.factor(nnettest_result))[[1]]
    accuracy_vec <- c(accuracy_vec, accuracy)
  }
  return(accuracy_vec)
}

system.time(cv(train_data,test_data))

result = cv(train_data,test_data)
print(result)
# [1] 0.6833333 0.8111111 0.7944444 0.7500000 0.7455556

size = which.max(result) # best size for the train model = 2
# for running it: nnet_train(train_data, size)

### Accuracy & Error rate for neural net
# 0.8111111 with size 2
# -> Error rate: 0.1888889 %

### Running time for neural net
#user  system elapsed 
#14.571   0.020  14.600

