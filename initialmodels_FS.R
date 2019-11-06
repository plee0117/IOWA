library(readr)
library(dplyr)
library(ggplot2)
library(VIM)
library(mice)
library(car)
library(glmnet)

train  = read.csv("train.csv")

train = train %>% mutate('psqftGla' = SalePrice/GrLivArea)



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

train_red = train %>% select(-nacols) #remove na columns

vars_name <- train_red %>%  select(-SalePrice) %>% select_if(is.factor) %>% colnames() %>%  paste(collapse = "+")  
model_string <- paste("y  ~",vars_name )


#split test and train (both within train data set)
set.seed(0)
trainrows = sample(1:nrow(train), 0.8*nrow(train))
x <- model.matrix(as.formula(model_string), train_red)
testrows = (-trainrows)
y = train$SalePrice

#lasso regression

grid = 10^seq(5, -2, length = 100)

lasso1 = cv.glmnet(x = x[trainrows,], y = train_red$SalePrice[trainrows], alpha = 1, lambda = grid, nfolds = 10)

lasso1_predict = predict.cv.glmnet(lasso1, s=lasso1$lambda.min, newx = x[testrows, ])

mse_lasso = mean((lasso1_predict - y[testrows])^2)


#view coeffiecnts with lambda min
predict(lasso1, s = lasso1$lambda.min, type = "coefficients")

sst_lasso = sum((y[testrows] - mean(y[testrows]))^2)
sse_lasso = sum((lasso1_predict - y[testrows])^2)

rsq_lasso= 1 - sse_lasso / sst_lasso
rsq_lasso #0.7632654



#RIDGE ######################################

ridge1 = cv.glmnet(x = x[trainrows,], y = train_red$SalePrice[trainrows], alpha = 0, lambda = grid, nfolds = 10)
ridge1_predict = predict.cv.glmnet(ridge1, s=lasso1$lambda.min, newx = x[testrows, ])
mse_ridge = mean((ridge1_predict - y[testrows])^2)

predict(ridge1, s = ridge1$lambda.min, type = "coefficients")
ridge


sst_ridge = sum((y[testrows] - mean(y[testrows]))^2)
sse_ridge = sum((ridge1_predict - y[testrows])^2)

rsq_ridge= 1 - sse_ridge / sst_ridge
rsq_ridge # 0.7716372

