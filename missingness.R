################################################
# Missingness Analysis - Iowa training dataset #
# Paul Lee, Bettina Meier  Oct 2019           #
################################################

# 1. Load libraries and dataset ####
library(mice)
library(VIM)
library(dplyr)

train = read.csv('train.csv')
head(train)
dim(train)
str(train)
colnames(train)
View(train) # NaNs and empty cells

# 2. Check missing values ####
sum(is.na(train)) # 6965 NAs in training dataset
VIM::aggr(train, numbers = T, prob = c(T,F))
md.pattern(train, rotate.names = T)

names = colnames(train[colSums(is.na(train))>0])
length(names) # 19 columns contain NAs

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

# Alley 93.7% missing
train %>% filter(!is.na(Alley)) %>% select(Alley) %>% group_by(Alley) %>% count()
# 50 Gravel, 41 Pavement, 93.7% NA = no Alley access 

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
# using MSZoning 229/259 properties with missing LotFrontage information are in RL, Residential Low Density areas
# The purpose of this zone is to create a living environment of high standard for primarily single-family dwellings
# Front Yard. There shall be a front yard having a minimum depth of not less than 20 feet, except that for a cul-de-sac or knuckle lot
# the minimum setback shall be not less than 10 feet.

train %>% filter(is.na(LotFrontage)) %>% select(MSZoning, PavedDrive) %>% group_by(MSZoning, PavedDrive) %>% count()
# All properties with NAs for LotFrontage have a Driveway, Y Paved, P Partially Paved, N Dirt/Gravel, offset from road?
# impute 0? Correlated with LotArea for those that are not zero?

# check LotArea (Lot size in square feet), LotFrontage(Linear feet of street connected to property), LotShape
train %>% filter(!is.na(LotFrontage)) %>% select(LotFrontage, LotArea, LotShape) %>% group_by(LotFrontage, LotArea, LotShape) %>% count()
# create a ratio? Hm, LotFrontage depends on LotArea, but Lots also have different shapes.
# change possibly into: on street, off-set from street?

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

# Missing Mas* are missing together but doesn't implying the lack of masonry; only 8 instances, maybe MCAR, can be ignored?
# Only 7 houses have pools; 3 Quality groups; Both columns (PoolArea and PoolQC) can be ignored
# Only 54 houses with MiscFeature; 49 with sheds 2 with 2nd garage, 2 unidentified, 1 Tennis Court; can be ignored
# Fence NA should be converted to 'No Fence'; still questionable whether it should be used with 80% without fences
# Alley NA should be converted to 'No Alley'; possibly should be ignored with 94% without alley
# FireplaceQu NA implies no fire place in property; All properties with fireplaces have a quality listed; replace with 'No Fireplace'; keep
# Garage* are NA together implying the lack of garage, should be replaced with 'No Garage'; keep
# Bsmt* are NA together except 1 unfinished basement MCAR, implying the lack of basement, should be replaced with 'No Basement'; keep
# Electrical missing possibly MCAR, the 1 missing value can be ignored or imputed with 'SBrkr' since 91% are of this type, others are usually in houses built before 1960, this one was 2006


# 5. Impute missing values ####

# DONE IN PYTHON



class(names)
names
names_reduced = names[c(-10, -11,-17,-19)]


md.pattern(train[names])
md.pattern(train[c('Fence','Alley')])
train %>% group_by(Alley) %>% count()
train %>% group_by(Fence) %>% count()

table(train['Fence'],train['Alley'])
