#######################################
# install & load necessary packages
#######################################
#install.packages("gbm")
library(gbm)
library(xgboost)
library(caret)
library(plyr)
library(dplyr)
library(e1071)
#######################################
# Load sift features + labels
#######################################
# setwd("C:/Users/enriquethemoist/Dropbox/Columbia Work/4A - one/Applied Data Science/Project 3/training set")
# sift_train = read_csv("sift_train.csv")
# label = read_csv("label_train.csv")
# data = data.frame(label[,2], sift_train[,2:ncol(sift_train)])
# colnames(data)[1] = "label"
# 
# #######################################
# # divide into train & test (70:30)
# #######################################
# set.seed(123)
# index = sample(1:nrow(data), size=0.7*nrow(data))
# train_data = data[index,]
# test_data = data[-index,]

#dat_train = training features
#label_train = labels 
#K = number of folds
#d = a certain interaction depth
cv.function <- function(dat_train, label_train, d, K){
  
  library(gbm)
  
  n <- length(label_train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- dat_train[s != i,]
    train.label <- label_train[s != i]
    test.data <- dat_train[s == i,]
    test.label <- label_train[s == i]
    
    par <- list(depth=d)
    fit <- gbm.fit(x=dat_train, y=label_train,
                   n.trees=250,
                   distribution = "multinomial",
                   interaction.depth=par$depth,
                   bag.fraction = 0.5,
                   verbose=FALSE)
    best_iter <- gbm.perf(fit, method="OOB")
    fit.gbm<-list(fit=fit,iter=best_iter)
    pred <- predict(fit.gbm$fit, newdata=test_data,n.trees=fit.gbm$iter,type="response")  
    pred<-as.numeric(pred>0.5)
    cv.error[i] <- mean(pred != test.label)  
    
  }			
  return(c(mean(cv.error),sd(cv.error)))
  
}




gbm_train<-function(dat_train,label_train) {
  depth_values <- c(seq(3,15,3))
  err_cv <- array(dim=c(length(depth_values), 2))
  K <- 5  
  for(k in 1:length(depth_values)){
    cat("k=", k, "\n")
    err_cv[k,] <- cv.function(dat_train, label_train, depth_values[k], K)
  }
  save(err_cv, file="C:/Users/enriquethemoist/Dropbox/Columbia Work/4A - one/Applied Data Science/Project 3/training set/baseline__train_error.RData")
  
  depth_best <- depth_values[which.min(err_cv[,1])]
  par_best <- list(depth=depth_best)
  
  
  fit_gbm <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=250,
                     distribution="multinomial",
                     interaction.depth=par_best$depth,
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB")
  fit_train<-list(fit=fit_gbm, iter=best_iter)
  save(fit_train, file="C:/Users/enriquethemoist/Dropbox/Columbia Work/4A - one/Applied Data Science/Project 3/training set/baseline__train_fit.RData")
  print(err_cv)
  print(depth_best)
  return(fit_train)
  
}

#system.time(result<-gbm_train(train_data[,2:ncol(train_data)],train_data$label))
