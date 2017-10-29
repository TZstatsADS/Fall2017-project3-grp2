library(gbm)
library(caret)
library(DMwR)
library(nnet)
library(randomForest)
library(e1071)

### Random Forest Train Start ##
randForest_train <- function(train_data, n_trees) {
  # train_data is a matrix with the first column as "label", each row is a data point
  # Hyperparameter n_trees
  
  train_data$label<- as.factor(train_data$label)
  label_col_ind <- which(colnames(train_data)=="label")
  bestmtrlabel <- tuneRF(y = train_data$label, x= train_data[,-label_col_ind], stepFactor=1.5, improve=1e-5, ntree=n_trees)
  best_mtrlabel <- bestmtrlabel[,1][which.min(bestmtrlabel[,2])]
  
  
  randForest_model <- randomForest(label ~ ., data = train_data, ntree=n_trees, mtry=best_mtrlabel, importance=T)
  return(randForest_model)
}
### Random Forest Train End ##


### Random Forest Test Start ##
randForest_test <- function(rf_model, test_data) {
  # rf_model is the fitted random forest model
  # test_data has no "label" column, each row is a data point
  
  return(predict(rf_model, test_data, type = "class"))
}
### Random Forest Test End ##


## Random Forest CV Start ##
randForest_cv <- function(train_data, n_trees_vec, K) { 
  # train_data is a matrix with the first column as "label", each row is a data point
  # n_trees_vec is a vector of numbers of n_trees
  # K is the number of folds
  
  
  n <- nrow(train_data)
  n_fold <- floor(n/K)
  fold <- sample(rep(1:K, c(rep(n_fold, K-1), n-(K-1)*n_fold)))  
  cv_error <- matrix(NA, nrow = K, ncol = length(n_trees_vec))
  running_time <- matrix(NA, nrow =  K, ncol = length(n_trees_vec))
  
  for (i in 1:K) {
    infn_train_data <- train_data[fold != i,]
    
    infn_test_data <- train_data[fold == i,-1]
    infn_test_label <- train_data[fold == i,1]
    
    for (j in 1:length(n_trees_vec)) {
      running_time[i,j] <- system.time(fit <- randForest_train(infn_train_data, n_trees = n_trees_vec[j]))[3]
      pred <- randForest_test(fit, infn_test_data)  
      cv_error[i, j] <- mean(pred != infn_test_label)
      
      #browser()
    }
    
  }
  #cv_error
  CVmisses <- apply(cv_error, 2, mean)
  Running_Time <- apply(running_time, 2, mean)
  data.frame(n_tree = n_trees_vec, cv_error = CVmisses, running_time = Running_Time)
  
  
}  
## Random Forest CV End ##