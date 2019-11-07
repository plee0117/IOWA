library(readr)
library(dplyr)
library(ggplot2)
library(VIM)
library(mice)
library(car)
library(glmnet)
library(caret)

train  = read.csv("train_clean.csv")

#train = train %>% mutate('psqftGla' = SalePrice/GrLivArea)

realtest = read.csv("test.csv")


#LINEAR ######################################
model1 = lm(psqftGla ~ Neighborhood + OverallQual + OverallCond 
            + YearRemodAdd + MSSubClass + TotRmsAbvGrd + 
              SaleType + SaleCondition + MoSold + Condition1 + 
              ExterQual, data = train)

summary(model1)  #Multiple R-squared:  0.6496,	Adjusted R-squared:  0.6361 

vif(model1)


######################

model2 = lm(SalePrice ~ Neighborhood + OverallQual + OverallCond 
            + YearRemodAdd + MSSubClass + TotRmsAbvGrd  + SaleCondition + MoSold + Condition1 + 
              ExterQual + GrLivArea, data = train)
summary(model2) #Multiple R-squared:  0.8271,	Adjusted R-squared:  0.8213 

vif(model2)


#LASSO ######################################


nacols = colnames(train)[unlist(lapply(train, function(x) any(is.na(x)))) == TRUE] #find columns with nas

train_red = train %>% select(-nacols, -psqftGla) #remove na columns
realtest_red = realtest %>% select(-nacols)

vars_name <- train_red %>%  select(-SalePrice) %>% select_if(is.factor) %>% colnames() %>%  paste(collapse = "+")  
model_string <- paste("y  ~",vars_name )

vars_name_test <- realtest_red %>% select_if(is.factor) %>% colnames() %>%  paste(collapse = "+")  
model_string_test <- paste("y  ~",vars_name_test )



#split test and train (both within train data set)
set.seed(0)
trainrows = sample(1:nrow(train), 0.8*nrow(train))
x <- model.matrix(as.formula(model_string), train_red)
testrows = (-trainrows)
y = log(train$SalePrice)

x_realtest = model.matrix(as.formula(model_string_test), realtest_red)

#lasso regression

grid = 10^seq(5, -2, length = 100)

lasso1 = cv.glmnet(x = x[trainrows,], y = log(train_red$SalePrice[trainrows]), alpha = 1, lambda = grid, nfolds = 10)

lasso1_predict = predict.cv.glmnet(lasso1, s=lasso1$lambda.min, newx = x[testrows, ])

mse_lasso = mean((lasso1_predict - y[testrows])^2)
sqrt(mse_lasso) #[1] 0.212773 with na cols dropped


#view coeffiecnts with lambda min
predict(lasso1, s = lasso1$lambda.min, type = "coefficients")

sst_lasso = sum((y[testrows] - mean(y[testrows]))^2)
sse_lasso = sum((lasso1_predict - y[testrows])^2)

rsq_lasso= 1 - sse_lasso / sst_lasso
rsq_lasso #0.7934248 on clean data



#RIDGE ######################################

ridge1 = cv.glmnet(x = x[trainrows,], y = log(train_red$SalePrice[trainrows]), alpha = 0, lambda = grid, nfolds = 10)
ridge1_predict = predict.cv.glmnet(ridge1, s=lasso1$lambda.min, newx = x[testrows, ])

mse_ridge = mean((ridge1_predict - y[testrows])^2)
sqrt(mse_ridge) #[1] 0.1962498 with na col dropped

sst_ridge = sum((y[testrows] - mean(y[testrows]))^2)
sse_ridge = sum((ridge1_predict - y[testrows])^2)

rsq_ridge= 1 - sse_ridge / sst_ridge
rsq_ridge # 0.7849663 on clean data

#predict(ridge1, s = lasso1$lambda.min, type = "coefficients")

#testy_ridge1 =predict.cv.glmnet(ridge1, s=lasso1$lambda.min, newx = actualtest[testrows, ])

###########

sqrt(nrow(x[trainrows,])) #34

knnfit = knnreg(x[trainrows,],y = log(train_red$SalePrice[trainrows]), k = 34)
knnfit

knn_predicted = predict(knnfit, x[testrows,])

mse_knn = mean((knn_predicted - y[testrows])^2)
sqrt(mse_knn) #[1] 0.1962498 with na col dropped

sst_knn = sum((y[testrows] - mean(y[testrows]))^2)
sse_knn = sum((knn_predicted - y[testrows])^2)

rsq_knn= 1 - sse_knn / sst_knn
rsq_knn 


