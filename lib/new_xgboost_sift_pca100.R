###################################
#### XGBoost with SIFT+PCA_100 ####
###################################

# Load libraries
library(xgboost)
library(caret)
library(plyr)
library(dplyr)
library(e1071)

# 
# # Use sift features
# sift_pca<-read.csv("feature_pca100.csv",header = T, as.is = T)
# label<-read.csv("label_train.csv",header = T,as.is = T)
# dat<-cbind(label[,2],sift_pca[,-1])
# colnames(dat)[1]<-"label"
# 
# set.seed(500)
# # Train and test split
# train_index<-sample(1:nrow(dat),0.7*nrow(dat))
# 
# xgb_variables<-as.matrix(dat[,-1]) # Full dataset
# xgb_label<-dat[,1] # Full label
# 
# # Split train data
# xgb_train<-xgb_variables[train_index,]
# train_label<-xgb_label[train_index]
# train_matrix<-xgb.DMatrix(data = xgb_train, label=train_label)
# 
# # Split test data
# xgb_test<-xgb_variables[-train_index,]
# test_label<-xgb_label[-train_index]
# test_matrix<-xgb.DMatrix(data = xgb_test, label=test_label)


# Basic model
basic = xgboost(data = train_matrix,
                max.depth=3,eta=0.01,nthread=2,nround=50,
                objective = "multi:softprob",
                eval_metric = "mlogloss",
                num_class = 3,
                verbose = F)

#pred = predict(basic, test_matrix)
#prediction<-matrix(pred,nrow = 3,ncol = length(pred)/3) %>%
#  t() %>%
#  data.frame() %>%
#  mutate(label=test_label+1,max_prob=max.col(.,"last"))

## confusion matrix of test set
#confusionMatrix(factor(prediction$label),factor(prediction$max_prob),mode = "everything")

#### Basic model###
## Accuracy: 71.89%
## Parameters: max.depth=3, eta=0.01, nthread=2, nround=50


# # Tune the model
# xgb_params_3 = list(objective="multi:softprob",
#                     eta = 0.01,
#                     max.depth = 3,
#                     eval_metric = "mlogloss",
#                     num_class = 3)
# 
# # fit the model with arbitrary parameters
# xgb_3 = xgboost(data = train_matrix, 
#                 params = xgb_params_3,
#                 nrounds = 100,
#                 verbose = F)
# 
# # cross validation
# xgb_cv_3 = xgb.cv(params = xgb_params_3,
#                   data = train_matrix, 
#                   nrounds = 100,
#                   nfold = 5,
#                   showsd = T,
#                   stratified = T,
#                   verbose = F,
#                   prediction = T)
# 
# # set up the cross validated hyper-parameter search
# xgb_grid_3 = expand.grid(nrounds=c(100,250,500),
#                          eta = c(1,0.1,0.01),
#                          max_depth = c(2,4,6,8,10),
#                          gamma=1,
#                          colsample_bytree=0.5,
#                          min_child_weight=2,
#                          subsample = 1)
# 
# # pack the training control parameters
# xgb_trcontrol_3 = trainControl(method = "cv",
#                                number = 5,
#                                verboseIter = T,
#                                returnData = F,
#                                returnResamp = "all",
#                                allowParallel = T)

# train the model for each parameter combination in the grid

#ptm <- proc.time() ## start the time
#xgb_train_3 = train(x=train_matrix, y=train_label,
#                    trControl = xgb_trcontrol_3,
#                    tuneGrid = xgb_grid_3,
#                    method = "xgbTree")

#ptm2 <- proc.time()
#ptm2- ptm ## stop the clock

## Time for training: 350.92s


#head(xgb_train_3$results[with(xgb_train_3$results,order(RMSE)),],5)
# get the best model's parameters
#xgb_train_3$bestTune

# best model
#bst = xgboost(data=train_matrix,max.depth=4,eta=0.1,nthread=2,nround=250,colsample_bytree=0.5,min_child_weight=2,subsample=1,objective="multi:softprob",eval_metric="mlogloss",num_class=3)

#pred = predict(bst, test_matrix)
#prediction<-matrix(pred,nrow = 3,ncol = length(pred)/3) %>%
#  t() %>%
#  data.frame() %>%
#  mutate(label=test_label+1,max_prob=max.col(.,"last"))

## confusion matrix of test set
#confusionMatrix(factor(prediction$label),factor(prediction$max_prob),mode = "everything")

## Accuracy: 82.67%
## Parameters: max.depth=4, eta=0.1, nthread=2, nround=250