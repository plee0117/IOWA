import pandas as 

def deal_with_NAs(the_df):
    
    # Lot Frontage isn't fixed
    
    the_df = the_df.drop(columns=['MasVnrType', 'MasVnrArea', 'PoolArea', 'PoolQC', 'MiscFeature'])
    the_df.Fence.fillna(value = 'No Fence',inplace = True)
    the_df.Alley.fillna(value = 'No Alley',inplace = True)
    the_df.FireplaceQu.fillna(value = 'No Fireplace',inplace = True)
    the_df.Electrical.fillna(value = the_df['Electrical'].mode()[0], inplace = True)
    the_df[['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']] = \
    the_df[['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']].fillna(value = 'No Garage')
    the_df[["BsmtQual","BsmtCond","BsmtFinType1","BsmtExposure","BsmtFinType2"]] = \
    the_df[["BsmtQual","BsmtCond","BsmtFinType1","BsmtExposure","BsmtFinType2"]].fillna(value = 'No Basement')
    the_df.loc[(the_df["BsmtExposure"] == 'No Basement') & (the_df['BsmtCond']!='No Basement'),['BsmtExposure']] = the_df["BsmtExposure"].mode()[0]
    the_df.loc[(the_df["BsmtFinType2"] == 'No Basement') & (the_df['BsmtCond']!='No Basement'),["BsmtFinType2"]] = the_df["BsmtFinType2"].mode()[0]
    return the_df