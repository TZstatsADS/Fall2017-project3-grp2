

pca_features <- function(data, n){
  # Input data is just the matrix of features (NO Labels or image indeces)
  # Input n is the number of features you'd like to keep
  
  pca=prcomp(data, center=TRUE, scale=TRUE);

  load <- pca$rotation[,1:n]
  save(load,file = "sift_pca_loading.rda")
  
  return(pca$x[,1:n])
  #returns reduced-dimensional feature matrix
}

# Sample Code to produce pca pracessed train_data
# sift <- sift_train1
# 

# data <- pca_features(sift, 100)
# 
# # selected data with labels
# pca_train_data <- cbind(data, label_train0[,2])
# colnames(pca_train_data)[ncol(pca_train_data)] <- "label"
# pca_train_data<-as.data.frame(pca_train_data)