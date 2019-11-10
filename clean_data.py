import pandas as 

def deal_with_NAs(full_set):
    '''
    Convert ordinal categorical features into numbers
    Convert nominal "NA's" into "No" string to give meaning
    Impute missing nominal values with mode
    Impute missing ordinal and numerical values with median
    '''
    categorical_columns = ['MSSubClass','MSZoning','MasVnrType','PoolQC','MiscFeature','Street','Alley','LotShape',\
                           'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',\
                           'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',\
                           'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',\
                           'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',\
                           'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','Fence',\
                           'SaleType','SaleCondition','MoSold','YrSold']

    numerical_columns = ['LotArea','OverallQual','MasVnrArea','PoolArea','OverallCond','YearBuilt','YearRemodAdd',\
                         'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',\
                         'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',\
                         'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',\
                         'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

    ordinal_columns = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',\
                       'GarageCond','PoolQC','BsmtExposure','LandSlope','Utilities','LotShape','Functional','Electrical',\
                      'GarageFinish','PavedDrive','Fence','BsmtFinType1','BsmtFinType2']

    nominal_columns = list(set(categorical_columns)-set(ordinal_columns))

    ordinal_nas = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',\
                   'GarageCond','PoolQC','BsmtExposure','GarageFinish','Fence','BsmtFinType1','BsmtFinType2']

    nominal_nas = ['Alley', 'GarageType', 'MiscFeature']

    q_to_q = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Av':3,'Mn':2,'No':1,'Gtl':2,'Mod':1,'Sev':0,\
              'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,\
              'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0,'SBrkr':4,\
              'FuseA':3,'FuseF':2,'FuseP':1,'Mix':0,'Fin':3,'RFn':2,'Unf':1,'Y':2,'P':1,'N':0,\
              'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2}

    # Convert ordinal feature values from qualitative to quantitative except Na's
    for col in ordinal_columns:
        for k_ in q_to_q.keys():
            full_set.loc[full_set[col] == k_ ,col] = q_to_q[k_]

    # Convert ordinal feature Na's to 0's
    for col in ordinal_nas:
        full_set.loc[full_set[col].isna(),col] = 0
        
    # Convert nominal Na's to No
    for col in nominal_nas:
        full_set.loc[col].fillna(value = 'No', inplace = True)

    # imputation of real Na's aka missing values

    nas = np.sum(full_set.isna()).reset_index()
    nas.columns = ['feature', 'NAs']
    nas.set_index('feature', inplace=True)
    nas = nas[nas['NAs']>0].sort_values('NAs',ascending=False)

    numerical_missing = list(set(nas.index) & (set(ordinal_columns) | set(numerical_columns)))

    nominal_missing = list(set(nas.index) & set(nominal_columns))

    # impute nominal features with mode
    for col in nominal_missing:
        mode_val = full_set[col].mode()[0]
        full_set[col].fillna(value = mode_val, inplace = True)

    # impute numerical or ordinal with median 
    for col in numerical_missing:
        med_val = full_set[col].median()
        full_set[col].fillna(value = med_val, inplace = True)

    return full_set

def change_features(full_set):
    '''
    Combine:
    3SsnPorch, EnclosedPorch, ScreenPorch, OpenPorchSF, and WoodDeckSF to PorchType
    PoolArea and PoolQC to PoolYN
    
    Convert:
    Fence to FenceYN
    LotFrontage to LotOnRoad

    Drop:
    GarageCars
    GrLivArea
    MiscFeature
    Alley
    TotalBsmtSF
    MSSubClass
    '''
    
    # Combine 3SsnPorch, EnclosedPorch, ScreenPorch, OpenPorchSF, and WoodDeckSF into 1 ordinal feature ranking the types
    full_set['PorchType'] = pd.DataFrame(5 if full_set['3SsnPorch'][i] > 0 else \
                                         4 if full_set['EnclosedPorch'][i] > 0 else \
                                         3 if full_set['ScreenPorch'] > 0 else \
                                         2 if full_set['OpenPorchSF'] > 0 else \
                                         1 if full_set['WoodDeckSF'] > 0 else 0 \
                                         for i in range(1,len(full_set.WoodDeckSF)))

    # Combine PoolArea and PoolQC to PoolYN
    full_set['PoolYN'] = pd.DataFrame(1 if full_set.PoolArea[i] > 0 else 0 for i in range(1,len(full_set.PoolArea)))
    full_set.drop(columns = ['PoolArea','PoolQC'], inplace = True)

    # Convert Fence to FenceYN
    full_set['FenceYN'] = pd.DataFrame(1 if full_set.Fence[i] > 0 else 0 for i in range(1,len(full_set.Fence)))
    full_set.drop(columns = 'Fence')

    # Convert LotFrontage to LotOnRoad
    full_set['LotOnRoad'] = pd.DataFrame(1 if full_set.LotFrontage[i] > 0 else 0 for i in range(1,len(full_set.LotFrontage)))
    full_set.drop(columns = 'LotFrontage', inplace = True)

    # Drop GarageCars since GarageCars and GarageArea give same information !!Check for overlap of area between the car #s!
    full_set.drop(columns = 'GarageCars')

    # Drop GrLivArea since it is the sum of 1stFlrSF and 2ndFlrSF
    full_set.drop(columns = 'GrLivArea')

    # Drop MiscFeature since it has the same information as the MiscVal
    full_set.drop(columns = 'MiscFeature')

    # Drop Alley since it's a rare occurence !! Check Alley vs Price boxplot!!
    full_set.drop(columns = 'Alley')

    # Drop TotalBsmtSF since it's equal to sum of BsmtFinSF1,BsmtFinSF2, and BsmtUnfSF
    full_set.drop(columns = 'TotalBsmtSF')

    # Drop MSSubClass since it is a combination of YearBuilt, BldgType, and HouseStyle
    full_set.drop(columns = 'MSSubClass')

    return full_set
