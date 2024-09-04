#### ANALYSIS SKRIPT ####
# "To Follow or Not to Follow: Estimating Political Opinion From Twitter Data Using a Network-Based Machine Learning Approach"
# DATE: 29.08.2024 
# Author: Cornelia Sindermann

##############################
########## PACKAGES ##########
##############################
#library(readxl) # import excel files (if needed)
library(dplyr) # data handling/manipulation
library(FactoMineR) # correspondence analysis
library(ggplot2) # needed for factoextra
library(factoextra) # correspeondence analysis (figures)
library(arrow) # read parquet.zip type file
Sys.setenv(LIBARROW_MINIMAL = "false"); install.packages('arrow', force = TRUE)
library(CAinterprTools) # nr. of dimensions
library(gtools) #smartbind
library(magrittr) #invert CA
library(MASS) #invert CA


##############################
######## READ DATASETS #######
##############################
### Training ###
df_train_p = arrow::read_parquet("/.../[name_training_data_set].gzip")
df_train = as.data.frame(df_train_p); rm(df_train_p) # given file type, df_train_p wasn't a normal dataframe before -> df_train is

### Validation ###
df_val_p = arrow::read_parquet("/.../[name_validation_data_set].gzip")
df_val = as.data.frame(df_val_p); rm(df_val_p) # given file type, df_val_p wasn't a normal dataframe before -> df_val_is

### Test ###
df_test_p = arrow::read_parquet("/.../[name_test_data_set].gzip")
df_test = as.data.frame(df_test_p); rm(df_test_p) # given file type, df_test_p wasn't a normal dataframe before -> df_test is


##############################
###### PREPARE DATASETS ######
##############################

### Remove Meta Data Columns and Those that are Incorrect From Each Data Set ###
df_ca_train = df_train %>% dplyr::select(-c("X", "follower_id", "Count", "add_fol", "dif", "X__index_level_0__", "MullerAltermatt.csv", "BgmLudwig.csv"))
df_ca_val = df_val %>% dplyr::select(-c("X", "follower_id", "Count", "add_fol", "dif", "X__index_level_0__", "MullerAltermatt.csv", "BgmLudwig.csv"))
df_ca_test = df_test %>% dplyr::select(-c("name", "id", "MullerAltermatt.txt", "BgmLudwig.txt")) 
# meta data columns removed; "MullerAltermatt.csv", "BgmLudwig.csv" are removed because "MullerAltermatt.csv" = swiss politician; "BgmLudwig.csv" = mayor of Vienna, Austria

##############################
####### MERGE DATASETS #######
##############################

### Merge Training and Validation Data Set ###
df_ca_merge_train_val = rbind(df_ca_train, df_ca_val)


### Merge Training and Test Data Set - Further Preparation Necessary ###

### Delete Columns from Test Set That are Only in There ###
## 1. make sure names of variables are in same format (no ".csv" or ".txt" in the names) ##
names(df_ca_train) = stringr::str_remove_all(names(df_ca_train), ".csv")
names(df_ca_test) = stringr::str_remove_all(names(df_ca_test), ".txt")

## 2. actual deletion of columns only included in test data set (do not delete columns from training data set because they are needed in line with validation data set; training data set needs to be the same across all analyses) ##
# 2.1 find columns only in test data set
names_train = c(names(df_ca_train))
names_test = c(names(df_ca_test))
only_in_test = setdiff(names_test, names_train)
# 2.2 actually delete columns from test data set that are only in there
df_ca_test = df_ca_test %>% dplyr::select(-one_of(only_in_test))


### Merge Training and Test Data Set - Actual Merging ###
## 1. order of columns in both data sets does not match --> use smartbind not rbind to join data frames
# 1.1 merging:
df_ca_test = as.data.frame(df_ca_test)
df_ca_merge_train_test = smartbind(df_ca_train, df_ca_test, fill = NA) 
# 1.2 test whether it worked properly
table(df_ca_merge_train_test$X_sanaeabdi); sum(is.na(df_ca_merge_train_test$X_sanaeabdi)) #163 NA --> PERFECT
# 1.3 replace NA with 0 (because it means "not following")
df_ca_merge_train_test = df_ca_merge_train_test %>% replace(is.na(.), 0)
str(df_ca_merge_train_test) # all looks good !
rm(names_test); rm(names_train); rm(only_in_test) # remove what is not needed anymore



##############################
########## ANALYSIS ##########
##############################

### Convert Data Sets into Table Format ###
## train_val:
dt_ca_merge_train_val <- as.table(as.matrix(df_ca_merge_train_val))
## train_test:
dt_ca_merge_train_test <- as.table(as.matrix(df_ca_merge_train_test))


### CA on Training + Validation Set ###
set.seed(42)
ca_train_val <- CA(dt_ca_merge_train_val, ncp = (min(ncol(dt_ca_merge_train_val)-1, 248407-1)), row.sup = 248408:276008, graph = F)
summary(ca_train_val)
print(ca_train_val)
#save(ca_train_val, file = "[file_name_ca_train_val].RData")
#load("/.../[file_name_ca_train_val].RData")


ca_train_val$eig # number of eigenvalues > 1 = none
fviz_eig(ca_train_val, choice = "eigenvalue") # "ellbow": 1 dimension or 3 dimensions

## row coordinates training + validation ##
df_coord_train_users = as.data.frame(ca_train_val$row$coord)
df_coord_users_train = cbind(df_train,df_coord_train_users)

df_coord_suppl_val_users = as.data.frame(ca_train_val$row.sup$coord)
df_coord_users_val = cbind(df_val,df_coord_suppl_val_users)

## column coordinates training + validation ##
df_coord_col_train = ca_train_val$col$coord
df_coord_col_val = ca_train_val$col$coord 
# must be the same because same columns in both cases (train and validation)
# from here, column coordinates of dimension 1 and 2 can be extracted



### CA on Training + Test Set ###
set.seed(42)
ca_train_test <- CA(dt_ca_merge_train_test, ncp = (min(ncol(dt_ca_merge_train_test)-1, 248407-1)), row.sup = 248408:248570, graph = F)
summary(ca_train_test)
print(ca_train_test)
#save(ca_train_test, file = "[file_name_ca_train_test].RData")
#load("/.../[file_name_ca_train_test].RData")


ca_train_test$eig # number of eigenvalues > 1 = none
fviz_eig(ca_train_test, choice = "eigenvalue") # "ellbow": 1 dimension or 3 dimensions

## row coordinates training + validation ##
df_coord_train_users_2 = as.data.frame(ca_train_val$row$coord)
df_coord_users_train_2 = cbind(df_train,df_coord_train_users)

df_coord_suppl_test_users = as.data.frame(ca_train_test$row.sup$coord)
df_coord_users_test = cbind(df_test,df_coord_suppl_test_users) 

## column coordinates training + validation ##
df_coord_col_train_2 = ca_train_val$col$coord
df_coord_col_test = ca_train_test$col$coord 
# must be the same because same columns in both cases (train and validation)




#############################################################################
######################### CHECKS - TRAINING DATA SET ########################
#############################################################################

### Invert Correspondence Analysis ###

## compute row and column profiles ##
row_coords = df_coord_users_train[, c("Dim 1", "Dim 2")]
col_coords = as.data.frame(df_coord_col_train[, c("Dim 1", "Dim 2")])
## compute the explained variances ##
explained_variances = ca_train_val$eig[,2]/100 
explained_variances12 = ca_train_val$eig[1:2,2]/100


pinv <- ginv(as.matrix(row_coords) %*% diag(as.vector(explained_variances12)) %*% t(as.matrix(col_coords)))
#save(pinv, file = "[file_name_pinv_train].RData")
#load("/.../[file_name_pinv_train].RData")
pinv = as.matrix(pinv)
pinv = t(pinv)

dim(df_ca_train); dim(pinv) # fit = perfect
df_ca_train[1:5, 1:5]; pinv[1:5, 1:5]
m_ca_train = as.matrix(df_ca_train); v_ca_train = c(m_ca_train)
v_pinv = c(pinv)

prop_cor <- function(y, yhat) {
  y <- round(plogis(y), 0) # inverse logit function to scale continuous values to range [0,1]
  yhat <- round(plogis(yhat), 0) # inverse logit function to scale continuous values to range [0,1]
  
  prop <- sum(y == yhat)/length(y) # length or product of dimensions? #RAVELING IS MISSING -> If matrix is transformed to vector -> lengthis ok
  print(paste('Proportion correctly predicted', prop))
}
#The function takes in three arguments - y, yhat, and m. The first two arguments are arrays or vectors representing the true and predicted values, respectively. The third argument m is a string indicating the type of model being used.
#In R, the round function is used to round the values to the nearest integer. The plogis function from the stats package is used to apply the inverse logit function to scale continuous values to the range [0, 1].
#Finally, the function computes the proportion of correctly predicted values by counting the number of elements in y and yhat that match and dividing by the total number of elements. The result is printed to the console.


### Prediction Statistics ###
prop_cor(v_ca_train, v_pinv)

caret::confusionMatrix(as.factor(round(plogis(v_ca_train),0)), as.factor(round(plogis(v_pinv),0))) #Accuracy, Balanced accuracy

ModelMetrics::mcc(as.vector(round(plogis(v_ca_train),0)), as.vector(round(plogis(v_pinv),0)), 0.5) #MCC

MLmetrics::F1_Score(as.factor(round(plogis(v_ca_train),0)), as.factor(round(plogis(v_pinv),0))) #F1

z0 = 1 - plogis(v_pinv)
z1 = plogis(v_pinv)
z = cbind(z0,z1)
class = factor(round(plogis(v_ca_train),0))
class = ifelse(class == 0, 1, 2)
DescTools::BrierScore(z, class) # brier score


## remove objects that will be newly created for other checks ##
rm(row_coords); rm(col_coords); rm(explained_variances); rm(explained_variances12); rm(pinv); rm(v_pinv)
rm(z0); rm(z1); rm(z); rm(class)

#############################################################################
######################## CHECKS - VALIDATION DATA SET #######################
#############################################################################

### Invert Correspondence Analysis ###

## compute row and column profiles ##
row_coords = df_coord_users_val[, c("Dim 1", "Dim 2")]
col_coords = as.data.frame(df_coord_col_val[, c("Dim 1", "Dim 2")])
## compute the explained variances ##
explained_variances = ca_train_val$eig[,2]/100 
explained_variances12 = ca_train_val$eig[1:2,2]/100

pinv_val <- ginv(as.matrix(row_coords) %*% diag(as.vector(explained_variances12)) %*% t(as.matrix(col_coords)))
#save(pinv_val, file = "[file_name_pinv_val].RData")
#load("/.../[file_name_pinv_val].RData")
pinv_val = as.matrix(pinv_val)
pinv_val = t(pinv_val)

dim(df_ca_val); dim(pinv_val) # fit = perfect
df_ca_val[1:5, 1:5]; pinv_val[1:5, 1:5]
m_ca_val = as.matrix(df_ca_val); v_ca_val = c(m_ca_val)
v_pinv_val = c(pinv_val)


### Prediction Statistics ###
prop_cor(v_ca_val, v_pinv_val)

caret::confusionMatrix(as.factor(round(plogis(v_ca_val),0)), as.factor(round(plogis(v_pinv_val),0))) #Accuracy, Balanced accuracy

ModelMetrics::mcc(as.vector(round(plogis(v_ca_val),0)), as.vector(round(plogis(v_pinv_val),0)), 0.5) #MCC

MLmetrics::F1_Score(as.factor(round(plogis(v_ca_val),0)), as.factor(round(plogis(v_pinv_val),0))) #F1


z0 = 1 - plogis(v_pinv_val)
z1 = plogis(v_pinv_val)
z = cbind(z0,z1)
class = factor(round(plogis(v_ca_val),0))
class = ifelse(class == 0, 1, 2)
DescTools::BrierScore(z, class) # brier score



## remove objects that will be newly created for other checks ##
rm(row_coords); rm(col_coords); rm(explained_variances); rm(explained_variances12); rm(pinv_val); rm(v_pinv_val)
rm(z0); rm(z1); rm(z); rm(class)



#############################################################################
########################### CHECKS - TEST DATA SET ##########################
#############################################################################
## note that CA on training data set is the same when implemented on training + validation and training + test; this is why the CA is only inverted on the training data set once ##

### Invert Correspondence Analysis ###

## compute row and column profiles ##
row_coords = df_coord_users_test[, c("Dim 1", "Dim 2")]
col_coords = as.data.frame(df_coord_col_test[, c("Dim 1", "Dim 2")])
## compute the explained variances ##
explained_variances = ca_train_test$eig[,2]/100 
explained_variances12 = ca_train_test$eig[1:2,2]/100

pinv_test <- ginv(as.matrix(row_coords[complete.cases(row_coords), ]) %*% diag(as.vector(explained_variances12)) %*% t(as.matrix(col_coords))) #coordinates not for all rows available --> exclude those from analyses
#save(pinv_test, file = "[file_name_pinv_test].RData")
#load("/.../[file_name_pinv_test].RData")
pinv_test = as.matrix(pinv_test)
pinv_test = t(pinv_test)

dim(df_ca_test); dim(pinv_test) # dimensions do not match because coordinates are not available for all # but why is second dimension 1427 not 1416 --> because in df_ca_test the missing columns are not yet inclued --> all good)
#need to include missing columns in df_ca_test or get new df_ca_test from merged data set to account for ordering issues
df_ca_test = df_ca_merge_train_test[248408:248570,]
dim(df_ca_test); dim(pinv_test)
#need to exclude rows that only have 0s
df_ca_test$rowsums = df_ca_test %>% rowSums()
df_ca_test = subset(df_ca_test, df_ca_test$rowsums != 0); df_ca_test = df_ca_test %>% dplyr::select(-rowsums)
dim(df_ca_test); dim(pinv_test)

m_ca_test = as.matrix(df_ca_test); v_ca_test = c(m_ca_test)
v_pinv_test = c(pinv_test)


### Prediction Statistics ###
prop_cor(v_ca_test, v_pinv_test)

caret::confusionMatrix(as.factor(round(plogis(v_ca_test),0)), as.factor(round(plogis(v_pinv_test),0))) #Accuracy, Balanced accuracy

ModelMetrics::mcc(as.vector(round(plogis(v_ca_test),0)), as.vector(round(plogis(v_pinv_test),0)), 0.5) #MCC

MLmetrics::F1_Score(as.factor(round(plogis(v_ca_test),0)), as.factor(round(plogis(v_pinv_test),0))) #F1


z0 = 1 - plogis(v_pinv_test)
z1 = plogis(v_pinv_test)
z = cbind(z0,z1)
class = factor(round(plogis(v_ca_test),0))
class = ifelse(class == 0, 1, 2)
DescTools::BrierScore(z, class) # brier score








