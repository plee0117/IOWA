################################################
# Missingness Analysis - Iowa training dataset #
# 4NN  Nov 2019                                #
################################################

# 1. Load libraries and dataset ####
library(mice)
library(VIM)
library(dplyr)
library(ggplot2)

train = read.csv('train.csv')
head(train)
dim(train)
str(train)
colnames(train)
View(train) # NaNs and empty cells

# 2. Check missing values ####
sum(is.na(train)) # 6965 NAs in training dataset
VIM::aggr(train, numbers = T, prob = c(T,F))

par(mar=c(0,2, 0, 2))
md.pattern(train, rotate.names = T)

names = colnames(train[colSums(is.na(train))>0])
length(names) # 19 columns contain NAs
sum(!complete.cases(train))
ncol(train) # 81

# 3. Analyse missing values ####

# Create table for colums with missing values with total numbers and percent of missingness
missings = data.frame(names,missing = colSums(is.na(train[names])))
nrow(train) # 1460 total observations
missings['percentage']=missings['missing']/nrow(train)*100
missings %>% arrange(desc(percentage))

# Investigate reason for missingness
# PoolQC 99.5% missing
train %>% select(contains('Pool')) %>% filter(!is.na(PoolQC)) # 8/1460 values
train %>% select(contains('Pool')) %>% filter(PoolArea>0) # only 8 properties with pool, these have 3 associated PoolQC categories

# MiscFeatures 96.3% missing
train %>% filter(!is.na(MiscFeature)) %>% select(MiscFeature) %>% group_by(MiscFeature) %>% 
  count() # 54 properties with MiscFeatures, 49 with a Shed, 2 with 2nd garage, 1 Tennis Court, 2 undefined
# how does this correlate with MiscValues?
train %>% dplyr::select(MiscFeature, MiscVal) %>% group_by(MiscFeature, MiscVal) %>% count()
# Groups:   MiscFeature, MiscVal [24]
#MiscFeature MiscVal     n
#<fct>         <int> <int>
#1 Gar2           8300     1
#2 Gar2          15500     1
#3 Othr              0     1
#4 Othr           3500     1
#5 Shed              0     1
#6 Shed             54     1
#7 Shed            350     1
#8 Shed            400    11
#9 Shed            450     4
#10 Shed           480     2
# … with 14 more rows


# Alley 93.7% missing
train %>% filter(!is.na(Alley)) %>% dplyr::select(Alley) %>% group_by(Alley) %>% count()
# 50 Gravel, 41 Pavement, 93.7% NA = no Alley access 

train %>% filter(is.na(Alley) & !is.factor(Alley))
sapply(train[names], is.factor)

# Fence 80.8% missing
train %>% filter(!is.na(Fence)) %>% select(Fence) %>% group_by(Fence) %>% count()  
# 4 different fence qualities (privacy/wood), NA means no fence 

# FireplaceQC 47.3%
train %>% filter(!is.na(FireplaceQu)) %>% select(FireplaceQu) %>% group_by(FireplaceQu) %>% count()
train %>% select(contains('fire')) %>% filter(Fireplaces == 0) # FireplaceQC NA is associated with Fireplaces = 0
train %>% select(contains('fire')) %>% filter(Fireplaces>0) 
train %>% filter(Fireplaces>0) %>% group_by(Fireplaces,FireplaceQu) %>% count() # 5 different quality scores for Fireplaces

# LotFrontage 17.7%, Linear feet of street connected to property
train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage) %>% group_by(LotFrontage) %>% count() %>% tail()
train %>% filter(is.na(LotFrontage)) %>% select(BldgType) %>% group_by(BldgType) %>% count()
# 226/259 of the properties with missing values for LotFrontage are 1Fam Homes, not multistory apartments
train %>% filter(is.na(LotFrontage)) %>% select(LotConfig) %>% group_by(LotConfig) %>% count()
# ~ half 134/259 of the properties with missing values for LotFrontage are on an inside lot
train %>% filter(is.na(LotFrontage)) %>% select(Neighborhood) %>% group_by(Neighborhood) %>% count()
# reasonable spread across Neigborhoods, no clear asssociation
train %>% filter(is.na(LotFrontage)) %>% select(MSSubClass) %>% group_by(MSSubClass) %>% count()
# 99 are category 20 1-STORY 1946 & NEWER ALL STYLES, 69 are category 60 2-STORY 1946 & NEWER, 20 are category 80 SPLIT OR MULTI-LEVEL, 20 category 120 1-STORY PUD (Planned Unit Development)
# no clear trend

train %>% filter(is.na(LotFrontage)) %>% select(MSZoning) %>% group_by(MSZoning) %>% count()
train %>% filter(LotFrontage == 0) %>% select(LotFrontage) %>% count()
# using MSZoning 229/259 properties with missing LotFrontage information are in RL, Residential Low Density areas

train %>% filter(!is.na(LotFrontage)) %>% select(MSZoning, BldgType, LotFrontage,LotConfig) %>% group_by(MSZoning, BldgType,LotConfig) %>%
  summarise(mean = mean(LotFrontage), median = median(LotFrontage), count = n())

train %>% filter(is.na(LotFrontage)) %>% select(MSZoning, BldgType, LotFrontage,LotConfig) %>% group_by(MSZoning, BldgType,LotConfig) %>%
  summarise(mean = mean(LotFrontage), median = median(LotFrontage), count = n())

train %>% filter(!is.na(LotFrontage)) %>% select(Neighborhood, LotFrontage, LotConfig) %>% group_by(Neighborhood, LotConfig) %>%
  summarise(mean = mean(LotFrontage), median = median(LotFrontage), count = n())

# requires imputing of values!!!

# check LotArea (Lot size in square feet), LotFrontage(Linear feet of street connected to property), LotShape
train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage, LotArea, LotShape) %>% group_by(LotFrontage, LotArea, LotShape) %>%
  count() %>% head(20)

train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage, LotArea, LotShape) %>% group_by(LotFrontage, LotArea, LotShape) %>%
  mutate(LotSide = sqrt(LotArea/1.62)) 
# create a ratio? Hm, LotFrontage depends on LotArea, but Lots also have different shapes.

train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage, LotArea, LotConfig, Neighborhood) %>% group_by(Neighborhood, LotConfig) %>%
  summarise(mean_LotFrontage = mean(LotFrontage), median_Frontage= median(LotFrontage), count = n()) %>% tail(20) 
#  ggplot(aes(x=Neighborhood, y=mean_Lot_Size)) +
#  geom_bar(stat="identity")

train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage, LotArea, LotConfig, Neighborhood) %>% group_by(Neighborhood, LotConfig) %>%
  summarise(mean_Lot_Size = mean(LotArea), median_Lot_Size = median(LotArea), count = n()) %>% filter(Neighborhood == 'NWAmes')

train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage, LotArea, LotConfig, Neighborhood) %>% group_by(Neighborhood, LotConfig) %>%
  summarise(mean = mean(LotFrontage), median = median(LotFrontage), count = n()) %>% 
  ggplot(aes(x=Neighborhood, y=mean)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage, LotArea, LotConfig, Neighborhood) %>% group_by(Neighborhood, LotConfig) %>%
  summarise(mean = mean(LotFrontage), median = median(LotFrontage), count = n()) %>% 
  ggplot(aes(x=Neighborhood, y=median)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage, LotConfig, Neighborhood) %>% group_by(Neighborhood, LotConfig) %>%
  mutate(LotFrontEstimate = mean(LotFrontage)) 

train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage, Neighborhood) %>% group_by(Neighborhood) %>%
  summarize(LotFrontEstimate = mean(LotFrontage)) %>% head(20)
# Impute LotFrontage by mean of LotFrontage grouped by Neighborhood and LotConfig


# Columns associated with GarageTypes have same number and percentage of missingness
train %>% filter(is.na(GarageType)) %>% group_by(GarageType,GarageCond,GarageYrBlt, GarageFinish, GarageQual, GarageCond) %>%
  count()
# If GarageType is NA, indicating no garage, all other quality information contains NA as well

# Columns associated with BasementFeatures have very similar number and percentage of missingness
train %>% filter(is.na(BsmtFinType1)) %>% select(BsmtCond,BsmtQual,BsmtExposure)
# If BsmtFinType1 is NA, indicating no Basement, other variables have NAs as well 
# BsmtExposure and BsmtFinType2 have 1 additional missing value each

# Masonry veneer type and Masonry veneer area have same number of missing values
train %>% group_by(MasVnrType) %>% count()
train %>% filter(is.na(MasVnrType)) %>% select(MasVnrArea)
train %>% filter(is.na(MasVnrArea))
# Masonry veneer type has a None type, not clear what NAs indicate. But NAs are correlated between area and type. 

# 1 Electrical observation with NA
train %>% group_by(Electrical) %>% count() %>% tail() # majority of others are SBrkr
class(train[1,'YearBuilt'])

train %>% group_by(Electrical) %>% summarise(min(YearBuilt))  # any information of use of the different electrical systems?
train %>% group_by(Electrical) %>% summarise(max(YearBuilt))  # looks like SBrkr were still used in newer houses
train %>% group_by(Electrical) %>% summarise(min(YearRemodAdd)) # no difference

# 4. Summary ####

# Missing NAs* are missing together but doesn't implying the lack of masonry; only 8 instances, maybe MCAR, can be ignored?
# Only 7 houses have pools; 3 Quality groups; Both columns (PoolArea and PoolQC) can be ignored
# Only 54 houses with MiscFeature; 49 with sheds 2 with 2nd garage, 2 unidentified, 1 Tennis Court; can be ignored
# Fence NA should be converted to 'No Fence'; still questionable whether it should be used with 80% without fences
# Alley NA should be converted to 'No Alley'; possibly should be ignored with 94% without alley
# FireplaceQu NA implies no fire place in property; All properties with fireplaces have a quality listed; replace with 'No Fireplace'; keep
# Garage* are NA together implying the lack of garage, should be replaced with 'No Garage'; keep
# Bsmt* are NA together except 1 unfinished basement MCAR, implying the lack of basement, should be replaced with 'No Basement'; keep
# Electrical missing possibly MCAR, the 1 missing value can be ignored or imputed with 'SBrkr' since 91% are of this type, others are usually in houses built before 1960, this one was 2006

# Impute LotFrontage with mean of LotFrontage of Neighborhood and LotConfig!

# 5. Impute missing values ####

# DONE IN PYTHON



