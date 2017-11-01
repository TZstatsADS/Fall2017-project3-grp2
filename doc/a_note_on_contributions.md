### Contribution Statement

Project 3

Team members: Yiran Li,  Qingyun Lu, Enrique Olivo, Saaya Yasuda, Qihang Li

Summary: 

[Contribution Statement] 

+ Yiran Li:
  + Designed the workflow, arranged meetings and delegated tasks.
  + Created the training, testing, and cross validation function for Random Forest. The cross validation function calculates
  and compares the cv errors and running time of various n_tree values. 
  + Produced the PCA function and sample code for dimension reduction to a user-specified number of features.
  + Ran cross-validation on Neural Net on PCA-processed SIFT, Random Forest with IBP, SIFT, and PCA-processed SIFT, resulting
  in classification accuracy of 70.7%, 70.4%, 74.7% and 79.3% respectively. 
  + Created main.rmd.
  
+ Qingyun Lu: 
  + Worked on HOG feature extraction method.
  + Worked on XGBoost method for 3 classes classification and tuned the model with grid search.
  + Tried XGBoost with SIFT + PCA, SIFT, HOG, IBP, GRAYSCALE, GRAYSCALE + SIFT + PCA.

+ Enrique Olivo:
  + Created the PowerPoint presentation
  + Worked on GBM Baseline Model function

+ Saaya Yasuda: 
  + Created the training and testing function for multinomial logistic regression. Since multinom() doesn't take in parameters, instead of cross validation, I used a step function to try versions with a fewer variables in x.
  + Created the training, testing, and cross validation function for neural net. For cross validation, I tried different values of the size parameter and picked the one that's most accurate.
  + Ran all of those functions on SIFT, HOG, and LBP features, measured the time, and reported the accuracy/error rate.
  + Worked on extracting the SIFT features with the original Matlab code.

+ Qihang Li: 
  + Got the SIFT features with original Matlab code
  + Extracted LBP features using Matlab
  + Extracted Grayscale features using R
  + help organized GitHub directory
