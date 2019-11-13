import pandas as pd
import numpy as np

def deal_with_NAs(full_set):
	'''
	Same as applying: ordinal_features, nominal_nas, impute_num_median, and impute_cat_mode
	Convert ordinal categorical features into numbers
	Convert nominal "NA's" into "No" string to give meaning
	Impute missing nominal values with mode
	Impute missing ordinal and numerical values with median
	'''
	categorical_columns = ['MSSubClass','MSZoning','MasVnrType','PoolQC','MiscFeature','Street','Alley','LotShape',
						   'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
						   'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
						   'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
						   'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
						   'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','Fence',
						   'SaleType','SaleCondition','MoSold','YrSold']

	numerical_columns = ['LotArea','OverallQual','MasVnrArea','PoolArea','OverallCond','YearBuilt','YearRemodAdd',
						 'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
						 'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
						 'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
						 'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

	ordinal_columns = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
					   'GarageCond','PoolQC','BsmtExposure','LandSlope','Utilities','LotShape','Functional','Electrical',
					  'GarageFinish','PavedDrive','Fence','BsmtFinType1','BsmtFinType2']

	nominal_columns = list(set(categorical_columns)-set(ordinal_columns))

	ordinal_nas = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
				   'GarageCond','PoolQC','BsmtExposure','GarageFinish','Fence','BsmtFinType1','BsmtFinType2']

	nominal_nas = ['Alley', 'GarageType', 'MiscFeature']

	q_to_q = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Av':3,'Mn':2,'No':1,'Gtl':2,'Mod':1,'Sev':0,
			  'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,
			  'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0,'SBrkr':4,
			  'FuseA':3,'FuseF':2,'FuseP':1,'Mix':0,'Fin':3,'RFn':2,'Unf':1,'Y':2,'P':1,'N':0,
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
		full_set[col].fillna(value = 'No', inplace = True)

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



def ordinal_features(full_set):
	'''
	Convert categorical into ordinal and deal with its Na's
	'''
	ordinal_columns = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
					   'GarageCond','PoolQC','BsmtExposure','LandSlope','Utilities','LotShape','Functional','Electrical',
					  'GarageFinish','PavedDrive','Fence','BsmtFinType1','BsmtFinType2']

	ordinal_nas = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
				   'GarageCond','PoolQC','BsmtExposure','GarageFinish','Fence','BsmtFinType1','BsmtFinType2']

	q_to_q = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Av':3,'Mn':2,'No':1,'Gtl':2,'Mod':1,'Sev':0,
			  'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,
			  'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0,'SBrkr':4,
			  'FuseA':3,'FuseF':2,'FuseP':1,'Mix':0,'Fin':3,'RFn':2,'Unf':1,'Y':2,'P':1,'N':0,
			  'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2}

	# Convert ordinal feature values from qualitative to quantitative except Na's
	for col in ordinal_columns:
		if col in full_set.columns:
			for k_ in q_to_q.keys():
				full_set.loc[full_set[col] == k_ ,col] = q_to_q[k_]
		else:
			continue

	# Convert ordinal feature Na's to 0's
	for col in ordinal_nas:
		if col in full_set.columns:
			full_set.loc[full_set[col].isna(),col] = 0
		else:
			continue

	return full_set




def nominal_nas(full_set):
	'''
	Convert the nominal Na's into No's
	'''
	nominal_nas = ['Alley', 'GarageType', 'MiscFeature']

	# Convert nominal Na's to No
	for col in nominal_nas:
		if col in full_set.columns:
			full_set[col].fillna(value = 'No', inplace = True)	
		else:
			continue

	return full_set



def impute_num_median(full_set):
	'''
	Deal with actual missing values aka Na's
	'''

	numerical_columns = ['LotArea','OverallQual','MasVnrArea','PoolArea','OverallCond','YearBuilt','YearRemodAdd',
						 'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
						 'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
						 'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
						 'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

	ordinal_columns = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
					   'GarageCond','PoolQC','BsmtExposure','LandSlope','Utilities','LotShape','Functional','Electrical',
					  'GarageFinish','PavedDrive','Fence','BsmtFinType1','BsmtFinType2']

	nas = np.sum(full_set.isna()).reset_index()
	nas.columns = ['feature', 'NAs']
	nas.set_index('feature', inplace=True)
	nas = nas[nas['NAs']>0].sort_values('NAs',ascending=False)

	numerical_missing = list(set(nas.index) & (set(ordinal_columns) | set(numerical_columns)))

	# impute numerical or ordinal with median 
	for col in numerical_missing:
		if col in full_set.columns:
			med_val = full_set[col].median()
			full_set[col].fillna(value = med_val, inplace = True)
		else:
			continue

	return full_set



def impute_cat_mode(full_set):
	'''
	Deal with actual missing values aka Na's
	'''

	categorical_columns = ['MSSubClass','MSZoning','MasVnrType','PoolQC','MiscFeature','Street','Alley','LotShape',
						   'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
						   'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
						   'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
						   'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
						   'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','Fence',
						   'SaleType','SaleCondition','MoSold','YrSold']

	ordinal_columns = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
					   'GarageCond','PoolQC','BsmtExposure','LandSlope','Utilities','LotShape','Functional','Electrical',
					  'GarageFinish','PavedDrive','Fence','BsmtFinType1','BsmtFinType2']

	nominal_columns = list(set(categorical_columns)-set(ordinal_columns))

	nas = np.sum(full_set.isna()).reset_index()
	nas.columns = ['feature', 'NAs']
	nas.set_index('feature', inplace=True)
	nas = nas[nas['NAs']>0].sort_values('NAs',ascending=False)

	nominal_missing = list(set(nas.index) & set(nominal_columns))

	# impute nominal features with mode
	for col in nominal_missing:
		if col in full_set.columns:
			mode_val = full_set[col].mode()[0]
			full_set[col].fillna(value = mode_val, inplace = True)
		else:
			continue

	return full_set



def change_features(full_set):
	'''
	Create:
	PropConsLot
	LivingArea
	ConstArea
	Yard
	YardPerc

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

	# Add new feature
	if all([x in full_set.columns for x in ['1stFlrSF','LotArea']]):
		full_set['PropConsLot'] = full_set['1stFlrSF'] / full_set['LotArea']

	if all([x in full_set.columns for x in ['GarageType']]):
		full_set['Garage'] = 0
		full_set['Garage'] = np.where((full_set['GarageType'] == 'BuiltIn'),1,full_set['Garage'])
		full_set['Garage'] = np.where((full_set['GarageType'] == 'Basment'),1,full_set['Garage'])
		full_set['Garage'] = np.where((full_set['GarageType'] == '2Types'),0.5,full_set['Garage'])

	if all([x in full_set.columns for x in ['TotalBsmtSF','GrLivArea','GarageArea','Garage']]):
		full_set['LivingArea'] = full_set['TotalBsmtSF'] + full_set['GrLivArea'] + full_set['GarageArea'] * full_set['Garage']

	if all([x in full_set.columns for x in ['1stFlrSF','GarageArea','Garage']]):
		full_set['ConstArea'] = full_set['1stFlrSF'] + full_set['GarageArea'] * full_set['Garage']

	if all([x in full_set.columns for x in ['GrLivArea','LotArea']]):  #!!!!!! should we keep this due to multicollinearity????
		full_set['ConstCost'] = full_set['GrLivArea'] * 100 + 4 * full_set['LotArea']

	if all([x in full_set.columns for x in ['LotArea','ConstArea']]):  #!!!!!! should we keep this due to multicollinearity????
		full_set['Yard'] = 0
		full_set['Yard'] = full_set['LotArea'] - full_set['ConstArea']

	if all([x in full_set.columns for x in ['ConstArea','LotArea']]):
		full_set['YardPerc'] = full_set['ConstArea'] / full_set['LotArea']

	# Combine 3SsnPorch, EnclosedPorch, ScreenPorch, OpenPorchSF, and WoodDeckSF into 1 ordinal feature ranking the types
	if all( x in full_set.columns for x in ['3SsnPorch','EnclosedPorch','ScreenPorch','OpenPorchSF','WoodDeckSF']):
		full_set['PorchType'] = pd.DataFrame(5 if full_set['3SsnPorch'][i] > 0 else 
										 4 if full_set['EnclosedPorch'][i] > 0 else 
										 3 if full_set['ScreenPorch'][i] > 0 else 
										 2 if full_set['OpenPorchSF'][i] > 0 else 
										 1 if full_set['WoodDeckSF'][i]> 0 else 0 
										 for i in range(1,len(full_set.WoodDeckSF)))
		full_set['PorchArea'] = (full_set['3SsnPorch'] + full_set['EnclosedPorch'] + full_set['ScreenPorch'] + 
			full_set['OpenPorchSF'] + full_set['WoodDeckSF'])
		full_set.drop(columns = ['3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'OpenPorchSF', 'WoodDeckSF'], inplace = True)

	# Combine PoolArea and PoolQC to PoolYN
	if 'PoolArea' in full_set.columns:
		full_set['PoolYN'] = pd.DataFrame(1 if full_set.PoolArea[i] > 0 else 0 for i in range(1,len(full_set.PoolArea)))
		full_set.drop(columns = 'PoolArea', inplace = True)
		if 'PoolQC' in full_set.columns:
			full_set.drop(columns = 'PoolQC', inplace = True)

	# Convert Fence to FenceYN
	if 'Fence' in full_set.columns:
		full_set['FenceYN'] = pd.DataFrame(1 if full_set.Fence[i] > 0 else 0 for i in range(1,len(full_set.Fence)))
		full_set.drop(columns = 'Fence', inplace = True)

	# Convert LotFrontage to LotOnRoad
	if 'LotFrontage' in full_set.columns:
		full_set['LotOnRoad'] = pd.DataFrame(1 if full_set.LotFrontage[i] > 0 else 0 for i in range(1,len(full_set.LotFrontage)))
		full_set.drop(columns = 'LotFrontage', inplace = True)

	# Drop GarageCars since GarageCars and GarageArea give same information !!Check for overlap of area between the car #s!
	if 'GarageCars' in full_set.columns:
		full_set.drop(columns = 'GarageCars', inplace = True)

	# Drop GrLivArea since it is the sum of 1stFlrSF and 2ndFlrSF
	if 'GrLivArea' in full_set.columns:
		full_set.drop(columns = 'GrLivArea', inplace = True)

	# Drop MiscFeature since it has the same information as the MiscVal
	if 'MiscFeature' in full_set.columns:
		full_set.drop(columns = 'MiscFeature', inplace = True)

	# Drop Alley since it's a rare occurence !! Check Alley vs Price boxplot!!
	if 'Alley' in full_set.columns:
		full_set.drop(columns = 'Alley', inplace = True)

	# Drop TotalBsmtSF since it's equal to sum of BsmtFinSF1,BsmtFinSF2, and BsmtUnfSF
	if 'TotalBsmtSF' in full_set.columns:
		full_set.drop(columns = 'TotalBsmtSF', inplace = True)

	# Drop MSSubClass since it is a combination of YearBuilt, BldgType, and HouseStyle
	if 'MSSubClass' in full_set.columns:
		full_set.drop(columns = 'MSSubClass', inplace = True)

	return full_set

def label_encode(full_set):
	'''
	Label encoding nominal features
	'''
	categorical_columns = []

	for col in full_set.columns:
		if full_set[col].dtype in ['object']:
			categorical_columns.append(col)

	ordinal_columns = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
					   'GarageCond','PoolQC','BsmtExposure','LandSlope','Utilities','LotShape','Functional','Electrical',
					  'GarageFinish','PavedDrive','Fence','BsmtFinType1','BsmtFinType2']

	nominal_columns = list(set(categorical_columns)-set(ordinal_columns))
	nominal_kept = set(nominal_columns) & set(full_set.columns)
	full_set = pd.get_dummies(full_set, prefix_sep = '_', columns=nominal_kept, drop_first = True)

	return full_set	


def boxcox_these(full_set):
	boxcoxing = ['ConstArea', 'LivingArea', 'YardPerc', 'LotArea', '2ndFlrSF', 'BsmtUnfSF', 'BsmtFinSF1', 'LowQualFinSF', '1stFlrSF', 
	'MiscVal', 'BsmtFinSF2', 'Yard', 'ConstCost', 'PropConsLot']
	boxcoxing = list(set(boxcoxing) & set(full_set.columns))
	for i in boxcoxing:
		full_set[[i]] = boxcox(full_set[[i]] + 1)[0]

	return full_set

