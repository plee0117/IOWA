library(mice)
train = read.csv('train.csv')
head(train)
md.pattern(train)

library(VIM)
aggr(train)

names = colnames(train[colSums(is.na(train))>0])
missings = data.frame(names,missing = colSums(is.na(train[names])))

missings['percentage']=missings['missing']/nrow(train)*100
missings %>% arrange(desc(percentage))
train.colnames
colnames(train)
library(dplyr)
train %>% select(contains('Pool')) %>% filter(!is.na(PoolQC))
train %>% select(contains('Pool')) %>% filter(PoolArea>0)

train %>% filter(!is.na(MiscFeature)) %>% select(MiscFeature) %>% group_by(MiscFeature) %>% 
  count()

train %>% filter(!is.na(Fence)) %>% select(Fence) %>% group_by(Fence) %>% count()

train %>% filter(!is.na(FireplaceQu)) %>% select(FireplaceQu) %>% group_by(FireplaceQu) %>% count()
train %>% select(contains('fire')) %>% filter(Fireplaces>0)

train %>% filter(Fireplaces>0) %>% group_by(Fireplaces,FireplaceQu) %>% count()

train %>% group_by(Electrical) %>% count()

train %>% filter(is.na(GarageType)) %>% group_by(GarageType,GarageCond,GarageYrBlt) %>%
  count()

train %>% filter(is.na(BsmtFinType2)) %>% select(BsmtCond,BsmtQual,BsmtFinType1)

train %>% filter(is.na(BsmtExposure)) %>% select(BsmtCond,BsmtQual,BsmtFinType1)

  select(BsmtFinType2,BsmtExposure) %>% group_by(BsmtFinType2,BsmtExposure) %>% count()

train %>% filter(is.na(BsmtExposure)) %>% select(BsmtCond,BsmtQual,BsmtFinType1)

train %>% group_by(MasVnrType) %>% count()

train %>% filter(is.na(MasVnrType)) %>% select(MasVnrArea)
train %>% filter(is.na(MasVnrArea))

# Missing Mas* are missing together but doesn't implying the lack of masonry; only 8 instances, maybe MCAR, can be ignored?
# Only 7 houses have pools; 3 Quality groups; Both columns (PoolArea and PoolQC) can be ignored
# Only 54 houses with MiscFeature; 49 with sheds 2 with 2nd garage, 2 unidentified, 1 Tennis Court; can be ignored
# Fence NA should be converted to 'No Fence'; still questionable whether it should be used with 80% without fences
# Alley NA should be converted to 'No Alley'; possibly should be ignored with 94% without alley
# FireplaceQu NA implies no fire place in property; All properties with fireplaces have a quality listed; replace with 'No Fireplace'; keep
# Garage* are NA together implying the lack of garage, should be replaced with 'No Garage'; keep
# Bsmt* are NA together except 1 unfinished basement MCAR, implying the lack of basement, should be replaced with 'No Basement'; keep
# Electrical missing possibly MCAR, the 1 missing value can be ignored or imputed with 'SBrkr' since 91% are of this type, others are usually in houses built before 1960, this one was 2006

class(names)
names
names_reduced = names[c(-10, -11,-17,-19)]

names_no_pool = names[-11]

md.pattern(train[names])
md.pattern(train[c('Fence','Alley')])
train %>% group_by(Alley) %>% count()
train %>% group_by(Fence) %>% count()

table(train['Fence'],train['Alley'])

train %>% group_by(Electrical) %>% summarise(min(YearRemodAdd))
class(train[1,'YearBuilt'])
