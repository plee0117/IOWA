###############################################
# Univariate Analysis - Iowa training dataset #
# Team 4NN Nov 2019                           #
###############################################

# 1. Load libraries and dataset ####
library(mice)
library(VIM)
library(dplyr)

train = read.csv('train.csv')
test = read.csv('test.csv')
head(train)
dim(train)
str(train)
colnames(train)
View(train) 


# Multiple plot function
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

sapply(train, is.factor)
sapply(train, is.numeric)

options(scipen=999) # disables scientific notification (10+5)

Price <- train %>%
  ggplot(aes(SalePrice)) +
  geom_histogram(binwidth = 10000, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$SalePrice), lwd = 2) +
  labs(title = "SalesPrice (Training Set)",
       x = "$",
       y = "freq") +
  theme_minimal()

max(train$SalePrice) # 755000

property_feet_on_street <- train %>%
  ggplot(aes(LotFrontage)) +
  geom_histogram(binwidth = 10, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$LotFrontage), lwd = 2) +
  labs(title = "LotFrontage (Linear feet of street connected to property)",
       x = "feet",
       y = "freq") +
  theme_minimal()


train_LotArea <- train %>%
  ggplot(aes(LotArea)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$LotArea), lwd = 2) +
  labs(title = "Train LotArea (Lot size in square feet)",
       x = "feet",
       y = "freq") +
  theme_minimal()


test_LotArea <- test %>%
  ggplot(aes(LotArea)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$LotArea), lwd = 2) +
  labs(title = "Test LotArea (Lot size in square feet)",
       x = "feet",
       y = "freq") +
  xlim(0, 220000) +
  theme_minimal()

max(train$LotArea)

multiplot(train_LotArea, test_LotArea, cols=1)

train_Property_Quality <- train %>%
  ggplot(aes(OverallQual)) +
  geom_histogram(binwidth = 1, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$OverallQual), lwd = 2) +
  labs(title = "Train OverallQual (Rates the \noverall material and finish of the house)",
       x = "rating 1:10 (very poor to very excellent)",
       y = "freq") +
  theme_minimal()

test_Property_Quality <- test %>%
  ggplot(aes(OverallQual)) +
  geom_histogram(binwidth = 1, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$OverallQual), lwd = 2) +
  labs(title = "Test OverallQual (Rates the \noverall material and finish of the house)",
       x = "rating 1:10 (very poor to very excellent)",
       y = "freq") +
  theme_minimal()


train_OverallCondition <- train %>%
  ggplot(aes(OverallCond)) +
  geom_histogram(binwidth = 1, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$OverallCond), lwd = 2) +
  labs(title = "Train OverallCond (Rates the \noverall condition of the house)",
       x = "rating 1:10 (very poor to very excellent)",
       y = "freq") +
  theme_minimal()

test_OverallCondition <- test %>%
  ggplot(aes(OverallCond)) +
  geom_histogram(binwidth = 1, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$OverallCond), lwd = 2) +
  labs(title = "Test OverallCond (Rates the \noverall condition of the house)",
       x = "rating 1:10 (very poor to very excellent)",
       y = "freq") +
  theme_minimal()

multiplot(train_Property_Quality, test_Property_Quality, train_OverallCondition, test_OverallCondition, cols=2)

train_GrLivArea <- train %>%
  ggplot(aes(GrLivArea)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$GrLivArea), lwd = 2) +
  labs(title = "Train GrLivArea",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

test_GrLivArea <- test %>%
  ggplot(aes(GrLivArea)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$GrLivArea), lwd = 2) +
  labs(title = "Test GrLivArea",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

multiplot(train_GrLivArea, test_GrLivArea, cols=1)

numerical_train <- train[which(sapply(train, is.numeric))] # 38
numerical_test <- test[which(sapply(test, is.numeric))] # 38


temp_train <- lapply(names(numerical_train),
                     function(x) qplot(get(x), data=numerical_train, xlab=x))

temp_test <- lapply(names(numerical_test),
                     function(x) qplot(get(x), data=numerical_test, xlab=x))

do.call(grid.arrange,temp_train[c(17,20:24)])
do.call(grid.arrange,temp_test[c(17,20:24)])


do.call(grid.arrange,temp_train[26:28])
do.call(grid.arrange,temp_test[26:28])

do.call(grid.arrange,temp_train[c(10:13, 18:19)])
do.call(grid.arrange,temp_test[c(10:13, 18:19)])

do.call(grid.arrange,temp_train[c(9)])
do.call(grid.arrange,temp_test[c(9)])
names(numerical_train)

do.call(grid.arrange,temp_train[c(17, 14:16)])
do.call(grid.arrange,temp_test[c(17, 14:16)])

which(numerical_train$LowQualFinSF > 0)
which(numerical_test$LowQualFinSF > 0)

numerical_train %>% filter(LowQualFinSF > 0) %>% 
  summarize(median = median(SalePrice)) # 131,500

numerical_train %>% filter(LowQualFinSF == 0) %>% 
  summarize(median = median(SalePrice)) # 163,945


# check how GrLivArea, X1stFlrSF, X2ndFlrSF and TotalBsmtSF correlate!
library(dplyr)
numerical_train %>% dplyr::select(.,GrLivArea, X1stFlrSF, X2ndFlrSF, TotalBsmtSF) %>% mutate(sum=rowSums(numerical_train[,c("X1stFlrSF", "X2ndFlrSF")]))
# GrLivArea = X1stFlrSF + X2ndFlrSF

train_MSSubclass <- train %>%
  ggplot(aes(MSSubClass)) +
  geom_bar() +
  labs(title = "Train MSSubClass (type of dwelling)",
       x = "", 
       y = "freq") +
  theme_minimal()

test_MSSubclass <- test %>%
  ggplot(aes(MSSubClass)) +
  geom_bar() +
  labs(title = "Test MSSubclass (type of dwelling)",
       x = "", 
       y = "freq") +
  theme_minimal()

train_MSZoning <- train %>%
  ggplot(aes(MSZoning)) +
  geom_bar() +
  labs(title = "Train MSZoning (zoning classification)",
       x = "", 
       y = "freq") +
  theme_minimal()

test_MSZoning <- test %>%
  ggplot(aes(MSZoning)) +
  geom_bar() +
  labs(title = "Test MSZoning (zoning classification)",
       x = "", 
       y = "freq") +
  theme_minimal()

multiplot(train_MSSubclass, test_MSSubclass, train_MSZoning, test_MSZoning, cols=2)


built_train <- train %>%
  ggplot(aes(YearBuilt)) +
  geom_histogram(binwidth = 10, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$YearBuilt), lwd = 2) +
  labs(title = "Train YearBuilt (Original construction date)",
       x = "",
       y = "freq") +
  theme_minimal()


built_test <- test %>%
  ggplot(aes(YearBuilt)) +
  geom_histogram(binwidth = 10, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$YearBuilt), lwd = 2) +
  labs(title = "Test YearBuilt (Original construction date)",
       x = "",
       y = "freq") +
  theme_minimal()



built_train <- train %>%
  ggplot(aes(YearBuilt)) +
  geom_histogram(binwidth = 10, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$YearBuilt), lwd = 2) +
  labs(title = "Train YearBuilt (Original construction date)",
       x = "",
       y = "freq") +
  theme_minimal()

remod_train <- train %>%
  ggplot(aes(YearBuilt)) +
  geom_histogram(binwidth = 10, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$YearRemodAdd), lwd = 2) +
  labs(title = "Train Remodel date (same as construction date if no remodeling or additions)",
       x = "",
       y = "freq") +
  theme_minimal()

remod_test <- test %>%
  ggplot(aes(YearBuilt)) +
  geom_histogram(binwidth = 10, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$YearRemodAdd), lwd = 2) +
  labs(title = "Test Remodel date (same as construction date if no remodeling or additions)",
       x = "",
       y = "freq") +
  theme_minimal()

multiplot(built_train, built_test, remod_train, remod_test, cols=2)

month_train <- train %>%
  ggplot(aes(MoSold)) +
  geom_histogram(binwidth = 10, color = "black",fill = "lightblue") +
  labs(title = "Train Month Sold",
       x = "",
       y = "freq") +
  theme_minimal()

month_test <- test %>%
  ggplot(aes(MoSold)) +
  geom_histogram(binwidth = 1, color = "black",fill = "lightblue") +
  labs(title = "Test Month Sold",
       x = "",
       y = "freq") +
  theme_minimal()

month_train <- train %>%
  ggplot(aes(MoSold)) +
  geom_histogram(binwidth = 1, color = "black",fill = "lightblue") +
  labs(title = "Train Month Sold",
       x = "",
       y = "freq") +
  theme_minimal()

year_train <- train %>%
  ggplot(aes(YrSold)) +
  geom_histogram(binwidth = 1, color = "black",fill = "lightblue") +
  labs(title = "Train Year Sold",
       x = "",
       y = "freq") +
  theme_minimal()

year_test <- test %>%
  ggplot(aes(YrSold)) +
  geom_histogram(binwidth = 1, color = "black",fill = "lightblue") +
  labs(title = "Test Year Sold",
       x = "",
       y = "freq") +
  theme_minimal()

multiplot(month_train, month_test, year_train, year_test, cols=2)


train_LotArea <- train %>%
  ggplot(aes(LotArea)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$LotArea), lwd = 2) +
  labs(title = "Train LotArea (Lot size in square feet)",
       x = "feet",
       y = "freq") +
  theme_minimal()

train_LotArea

test_LotArea <- test %>%
  ggplot(aes(LotArea)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$LotArea), lwd = 2) +
  labs(title = "Test LotArea (Lot size in square feet)",
       x = "feet",
       y = "freq") +
  xlim(0, 220000) +
  theme_minimal()

test_LotArea
max(train$LotArea)

train_LotFrontage <- train %>%
  ggplot(aes(LotFrontage)) +
  geom_histogram(binwidth = 10, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$LotFrontage), lwd = 2) +
  labs(title = "Train LotFrontage (Linear feet of street connected \nto property)",
       x = "feet",
       y = "freq") +
  theme_minimal()

test_LotFrontage <- test %>%
  ggplot(aes(LotFrontage)) +
  geom_histogram(binwidth = 10, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$LotFrontage), lwd = 2) +
  labs(title = "Test LotFrontage (Linear feet of street connected \nto property)",
       x = "feet",
       y = "freq") +
  theme_minimal()

multiplot(train_LotArea, test_LotArea, train_LotFrontage, test_LotFrontage,cols=2)


train_ScreenPorch <- train %>%
  ggplot(aes(ScreenPorch)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$ScreenPorch), lwd = 2) +
  labs(title = "Train ScreenPorch",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

test_ScreenPorch <- test %>%
  ggplot(aes(ScreenPorch)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$ScreenPorch), lwd = 2) +
  labs(title = "Test ScreenPorch",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

train_OpenPorchSF <- train %>%
  ggplot(aes(OpenPorchSF)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$OpenPorchSF), lwd = 2) +
  labs(title = "Train OpenPorchSF",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

test_OpenPorchSF <- test %>%
  ggplot(aes(OpenPorchSF)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$OpenPorchSF), lwd = 2) +
  labs(title = "Test OpenPorchSF",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

train_EnclosedPorch <- train %>%
  ggplot(aes(EnclosedPorch)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$EnclosedPorch), lwd = 2) +
  labs(title = "Train EnclosedPorch",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

test_EnclosedPorch <- test %>%
  ggplot(aes(EnclosedPorch)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$EnclosedPorch), lwd = 2) +
  labs(title = "Test EnclosedPorch",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

train_X3SsnPorch <- train %>%
  ggplot(aes(X3SsnPorch)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$X3SsnPorch), lwd = 2) +
  labs(title = "Train X3SsnPorch",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

test_X3SsnPorch <- test %>%
  ggplot(aes(X3SsnPorch)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$X3SsnPorch), lwd = 2) +
  labs(title = "Test X3SsnPorch",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

train_WoodDeckSF <- train %>%
  ggplot(aes(WoodDeckSF)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$WoodDeckSF), lwd = 2) +
  labs(title = "Train WoodDeckSF",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

test_WoodDeckSF <- test %>%
  ggplot(aes(WoodDeckSF)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$WoodDeckSF), lwd = 2) +
  labs(title = "Test WoodDeckSF",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

test$WoodDeckSF

multiplot(train_ScreenPorch, test_ScreenPorch, train_OpenPorchSF, test_OpenPorchSF, train_EnclosedPorch, test_EnclosedPorch,
          train_X3SsnPorch, test_X3SsnPorch, train_WoodDeckSF, test_WoodDeckSF, cols=5)



train_PoolArea <- train %>%
  ggplot(aes(PoolArea)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(train$PoolArea), lwd = 2) +
  labs(title = "Train PoolArea",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

test_PoolArea <- test %>%
  ggplot(aes(PoolArea)) +
  geom_histogram(binwidth = 100, color = "black",fill = "lightblue") +
  geom_vline(xintercept = mean(test$PoolArea), lwd = 2) +
  labs(title = "Test PoolArea",
       x = "sqfeet",
       y = "freq") +
  theme_minimal()

multiplot(train_PoolArea, test_PoolArea, cols=1)



# look at categorical features 
train %>% select(is.factor(train)) %>% group_by() %>% count()

categorical_train <- train[which(sapply(train, is.factor))] # 43
categorical_test <- test[which(sapply(test, is.factor))] # 43


categorical_train
temp_train <- lapply(names(categorical_train),
               function(x) qplot(get(x), data=categorical_train, geom="bar", xlab=x))

temp_test <- lapply(names(categorical_test),
                     function(x) qplot(get(x), data=categorical_test, geom="bar", xlab=x))

do.call(grid.arrange,temp_train[37:43])
do.call(grid.arrange,temp_test[37:43])



