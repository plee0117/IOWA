import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from scipy.stats import boxcox
from scipy.special import boxcox1p
from scipy.special import inv_boxcox
from scipy.stats import boxcox_normmax
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#from mlxtend.regressor import StackingCVRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import VotingRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
import matplotlib.pyplot as plt

import cluster_fun


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#FUNCTIONS ==============================================================================================

def evaluate_model(model, x, y, folds, lmbda):
    print("Accuracy: ",round(model.score(x, y),4))
    if lmbda == None:
        ypred = cross_val_predict(model, x, y, cv = folds, n_jobs = -1)
        print("RMSLE = ", round(np.sqrt(mean_squared_log_error(abs(ypred), y )),4) )
        scores   = -cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv = folds)
        print("Cross-Val Score =",round(np.mean(abs(scores**.5))))
    else:
        ypred = cross_val_predict(model, x, y, cv = folds, n_jobs = -1)
        #ypred = inv_boxcox(ypred, lmbda)
        ypred = ((ypred*lmbda)+1)**(1/lmbda)
        y = ((y*lmbda)+1)**(1/lmbda)
        #y = inv_boxcox(y, lmbda)
        print("RMSLE = ", round(np.sqrt(mean_squared_log_error(ypred, y )),4) )
        scores = -cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv = folds)
        print("Cross-Val Score =",round(np.mean(abs(scores**.5))))



#DATA LOADING ===========================================================================================

train = pd.read_csv('train.csv').set_index('Id')
test = pd.read_csv('test.csv').set_index('Id')
test['SalePrice'] = -1
df = pd.concat([train,test],axis=0)
#full_set.to_csv('full_set.csv')
#full_set = full_set[train.columns]

#DATA PROCESSING ========================================================================================

df['TotalBsmtSF'] = np.where((df['TotalBsmtSF'].isna()),0,df['TotalBsmtSF'])

df['MSZoning'] = np.where((df['MSZoning'].isna()),df['MSZoning'].mode(),df['MSZoning'])
df['Exterior1st'] = np.where((df['Exterior1st'].isna()),df['Exterior1st'].mode(),df['Exterior1st'])
df['SaleType'] = np.where((df['SaleType'].isna()),df['SaleType'].mode(),df['SaleType'])
df['GarageType'] = np.where((df['GarageType'].isna()),'None',df['GarageType'])



df['BsmtFinSF1'] = np.where((df['BsmtFinSF1'].isna()),0,df['BsmtFinSF1'])

df['BsmtFinSF1'] = np.where((df['BsmtFinSF1'].isna()),0,df['BsmtFinSF1'])
df['BsmtFinSF2'] = np.where((df['BsmtFinSF2'].isna()),0,df['BsmtFinSF2'])
df['BsmtUnfSF'] = np.where((df['BsmtUnfSF'].isna()),0,df['BsmtUnfSF'])
df['BsmtFullBath'] = np.where((df['BsmtFullBath'].isna()),0,df['BsmtFullBath'])
df['BsmtHalfBath'] = np.where((df['BsmtHalfBath'].isna()),0,df['BsmtHalfBath'])
df['MasVnrArea'] = np.where((df['MasVnrArea'].isna()),0,df['MasVnrArea'])
df['GarageCars'] = np.where((df['GarageCars'].isna()),0,df['GarageCars'])


df['GarageArea'] = np.where((df['GarageArea'].isna()),df.loc[(df['GarageType']=='Detchd'),'GarageArea'].mean(),df['GarageArea'])

df['Garage'] = 0
df['Garage'] = np.where((df['GarageType'] == 'BuiltIn'),1,df['Garage'])
df['Garage'] = np.where((df['GarageType'] == 'Basment'),1,df['Garage'])
df['Garage'] = np.where((df['GarageType'] == '2Types'),0.5,df['Garage'])

df['LivingArea'] = df['TotalBsmtSF']+df['GrLivArea']+df['GarageArea']*df['Garage']

df['ConstArea'] = df['1stFlrSF']+df['GarageArea']*df['Garage']

df['Yard'] = 0
df['Yard'] = df['LotArea']-df['ConstArea']

df['YardPerc'] = df['ConstArea']/df['LotArea']

df['Bathrooms'] = df['FullBath'] + df['HalfBath']

df['Pool'] = 0
df['Pool'] = np.where((df['PoolArea'] > 0), 0, 1)

df['ConstCost'] = df['GrLivArea'] * 100 + 4 * df['LotArea']

#Convert into string Categorical Columns
df.MSSubClass = df.MSSubClass.astype(str)
#df.OverallQual = df.OverallQual.astype(str)
#df.OverallCond = df.OverallCond.astype(str)
df.YrSold = df.YrSold.astype(str)
df.MoSold = df.MoSold.astype(str)


#cols_to_keep = ['MSSubClass','YearBuilt','YearRemodAdd','MSZoning','LotArea',
#            'Neighborhood','HouseStyle','OverallQual','Exterior1st','ExterQual',
#            'CentralAir','SaleType','SaleCondition','LivingArea','ConstArea',
#            'Yard','YardPerc','Bathrooms','Pool','ConstCost', 'MoSold','YrSold','SalePrice']

#Eliminated LotFrontage and GarageYrBuilt
cols_to_keep = ['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'Fence', 
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice', 'Garage',
       'LivingArea', 'ConstArea', 'YardPerc', 'Bathrooms', 'Pool',
       'ConstCost']





data = df[cols_to_keep]


#ADDING COUNT OF HOUSES SOLD/MONTH COLUMN ###########

moData = data[:1460].groupby(["YrSold", "MoSold"]).agg({'SalePrice' : ['count', 'mean']})
moData['moSalesCount'] = moData["SalePrice"]["count"]
moData['moSPMean'] = moData["SalePrice"]["mean"]

moData=moData.drop(moData.columns[1], axis=1)
moData=moData.drop(moData.columns[0], axis=1)
moData = moData.reset_index()
moData['date'] =list(map(lambda x,y: str(y)+"-"+str(x),map(lambda x: "0"+str(x) if int(x)<10 else str(x), moData['MoSold']),moData['YrSold']))

moData = moData[['date', 'moSalesCount']]

moData = moData.set_index("date").transpose().to_dict(orient='list')

for k, v in moData.items():
    moData[k] = v[0]

data['moCount'] =list(map(lambda x,y: str(y)+"-"+str(x),map(lambda x: "0"+str(x) if int(x)<10 else str(x), data['MoSold']),data['YrSold']))

data['moCount'] = data['moCount'].map(moData) #map values from dictionary of mounth counts


###########################


categorical = []
numerical = []

for col in data.columns:
    if data[col].dtype in ['object']:
        categorical.append(col)
    else:
        numerical.append(col)

#data.to_csv('data.csv')


#Boxcox 
for i in ['LotArea','ConstCost','YardPerc','LivingArea','ConstArea']:
    data[i] = boxcox(data[i])[0]

data= data.drop([31,411,524,534,592,677,689,804,1047,1299])
data= data.drop([363,390,496,633,729,770,899,969,1063,1182])



##########KMEANS ON NEIGHBORHOODS################

cateogrical_all =  categorical

trainData = data[:1460]
trainData['spgla'] = trainData.SalePrice/trainData.GrLivArea
data = data.reset_index()

kdf = cluster_fun.cluster(4, trainData[['Neighborhood', 'spgla']]) #n tot = 25
data = cluster_fun.kReplace(data, kdf, 'Neighborhood', 'spgla')

categorical.remove('Neighborhood')

#K MEANS WILL GIVE ERRORS IS THERE ARE NAS OR VALUES IN TEST THAT ARENT IN TRAIN


##############################

data = pd.get_dummies(data, prefix_sep = '_', columns=categorical, drop_first = True)
np.sum(data.isna())


yTr = data.SalePrice[data['SalePrice']>=0]
xTr = data[data['SalePrice']>=0].drop(['SalePrice'], axis=1)
xTe = data[data['SalePrice']==-1].drop(['SalePrice'], axis=1)
transf_yTr_lambda=None

transf_yTr = boxcox(yTr)
transf_yTr_lambda = boxcox(yTr, lmbda=transf_yTr_lambda)[1]
yTr = transf_yTr[0]

x = xTr[:]
y = yTr[:]

#####################################

print("GBoosting","-"*20)
boost = GradientBoostingRegressor(n_estimators = 500)
boost.fit(x, y)
evaluate_model(boost, x, y, 10, transf_yTr_lambda)



print("Ridge","-"*20)
ridge = linear_model.RidgeCV(   alphas=np.arange(0.05,4,0.01),
                                normalize=[True,False],
                                fit_intercept=[True,False])
ridge.fit(x, y)
best_alpha = ridge.alpha_
print("Best alpha = ",best_alpha)
ridge = linear_model.Ridge(ridge.alpha_,normalize=True)
ridge.fit(x, y)
evaluate_model(ridge, x, y, 5, transf_yTr_lambda)


xmo = xTr['moCount'][:]
ymo = y

lin = linear_model.LinearRegression()
lin.fit(xmo,ymo)
print(lin.score)



