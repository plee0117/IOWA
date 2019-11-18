############################################################
# Feature Analysis and Engineering - Iowa training dataset #
# Team 4NN Nov 2019                                        #
############################################################

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


# 2.Checking different features for feature selection ####

# 2.1 SaleCondition has weird Sales, i.e. some within family which will not be 'regular' market sales
train %>% dplyr::select(SaleCondition) %>% group_by(SaleCondition) %>% count()

#SaleCondition     n
#<fct>         <int>
# 1 Abnorml         101
# 2 AdjLand           4
# 3 Alloca           12
# 4 Family           20
# 5 Normal         1198
# 6 Partial         125
# Keep only Normal SaleCondition, Family sales and Foreclosures don't follow normal Price predictions

# 2.2 How do GrLivArea, X1stFlrSF, X2ndFlrSF and TotalBsmtSF correlate?
numerical_train %>% dplyr::select(.,GrLivArea, X1stFlrSF, X2ndFlrSF, TotalBsmtSF) %>% mutate(sum=rowSums(numerical_train[,c("X1stFlrSF", "X2ndFlrSF")]))
# GrLivArea = X1stFlrSF + X2ndFlrSF, TotalBsmtSF is separate

# 2.3 How do TotalBsmtSF and other values correlate? 
train %>% dplyr::select(.,TotalBsmtSF, BsmtUnfSF, BsmtFinSF1, BsmtFinSF2) %>% 
  mutate(sum=rowSums(train[,c( "BsmtUnfSF", "BsmtFinSF1", "BsmtFinSF2")])) %>% 
  mutate(sum == TotalBsmtSF)
# column TotalBsmtSF is sum of all the other predictors, drop?

# 2.4. MSSubClass is a weird column with numerically encoded features also provided by other columns
# Also is there a useful way to combine Year Built and Year Remodelled?
train %>% dplyr::select(MSSubClass, BldgType, HouseStyle, YearBuilt, YearRemodAdd) %>%
  mutate(HouseAge=(2019-YearBuilt))

train %>% dplyr::select(MSSubClass, BldgType, HouseStyle, YearBuilt, YearRemodAdd) %>%
  filter(BldgType == "TwnhsE" & MSSubClass !=120)
# Townhouses are all Planned Unit Development, no real additional information in this column
# from my understanding

train %>% dplyr::select(MSSubClass, BldgType, HouseStyle, YearBuilt, YearRemodAdd) %>%
  filter(BldgType != "TwnhsE" & MSSubClass %in% c(120, 150, 160, 180))
# Also category Twnhs and some 1FamHomes

train %>% dplyr::select(MSSubClass, BldgType, HouseStyle) %>%
  group_by(MSSubClass, BldgType, HouseStyle) %>% count()
# Disagreement between some of these, miscategorisation of MSSubClass??

# 2.5. Porches, 5 different 

# 2.4. Correlation of GarageArea and Number of Garages
train %>% filter(GarageCars > 0) %>% ggplot(aes(x=GarageArea, color=as.factor(GarageCars))) +
  geom_density()
# GarageAreas overlap, GarageCars = 4 has weird size distribution..... 



