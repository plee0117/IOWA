import pandas as pd
import numpy as np

def deal_with_NAs(full_set):
    '''
    Convert ordinal categorical features into numbers
    Convert nominal "NA's" into "No" string to give meaning
    Impute missing nominal values with mode
    Impute missing ordinal and numerical values with median
    '''
    full_set = fix_ext(full_set)
    full_set = ordinal_features(full_set)
    full_set = nominal_nas(full_set)
    full_set = impute_num_median(full_set)
    full_set = impute_cat_mode(full_set)
    
    return full_set


def fix_ext(full_set):
    '''
    Fix spelling in Exterior2 and change duplicates with Exterior1 with None
    '''
    # fix spelling in Exterior2
    Ex2_splchk = {'Brk Cmn':'BrkComm','CmentBd':'CemntBd'}
    asdf = pd.DataFrame(Ex2_splchk[full_set['Exterior2nd'][i]] if full_set['Exterior2nd'][i] 
                        in Ex2_splchk else full_set['Exterior2nd'][i] for i in range(1,1+len(full_set['Exterior2nd'])))
    asdf['Id'] = asdf.index + 1
    full_set['Exterior2nd'] = asdf.set_index('Id')
    
    # Change duplicates to None
    asdf = pd.DataFrame('None' if full_set['Exterior2nd'][i] == full_set['Exterior1st'][i]
                       else full_set['Exterior2nd'][i] for i in range(1,1 + len(full_set['Exterior2nd'])))
    asdf['Id'] = asdf.index + 1
    full_set['Exterior2nd'] = asdf.set_index('Id')
    
    return full_set


def ordinal_features(full_set):
    '''
    Convert categorical into ordinal and deal with its Na's
    R
    '''

    q_to_q = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Av':3,'Mn':2,'No':1,'Gtl':2,'Mod':1,'Sev':0,\
              'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,\
              'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0,'SBrkr':4,\
              'FuseA':3,'FuseF':2,'FuseP':1,'Mix':0,'Fin':3,'RFn':2,'Unf':1,'Y':2,'P':1,'N':0,\
              'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2}

    # Convert discrete feature to string
    full_set['MSSubClass'] = full_set['MSSubClass'].astype(str)
    full_set['YrSold'] = full_set['YrSold'].astype(str)
    full_set['MoSold'] = full_set['MoSold'].astype(str)
    
    # Convert ordinal feature values from qualitative to quantitative except Na's
    for col in ordinal_columns:
        if col in full_set.columns:
            for k_ in q_to_q.keys():
                full_set.loc[full_set[col] == k_ ,col] = q_to_q[k_]
        else:
            continue

    # Convert ordinal feature Na's to 0's
    for col in ordinal_columns:
        if col in full_set.columns:
            full_set.loc[full_set[col].isna(),col] = 0
        else:
            continue

    return full_set


def nominal_nas(full_set):
    '''
    Convert the nominal Na's into No's
    '''
    nominal_nas = ['Alley', 'GarageType', 'MiscFeature','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
                   'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC','BsmtExposure','GarageFinish',
                   'Fence','BsmtFinType1','BsmtFinType2']

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
    train_part = full_set[1:1461]

    import itertools
    n = set(full_set['Neighborhood'])
    l = set(full_set['LotConfig'])
    combinations = list(itertools.product(n,l))
    combinations
    imputation_dict = train_part[train_part['LotFrontage'].notnull()].groupby(['Neighborhood'
                                                            ,'LotConfig'])[['LotFrontage']].mean().round(2).to_dict()
    for key in combinations:
        if key in imputation_dict['LotFrontage'].keys():
            pass
        else:
            imputation_dict['LotFrontage'][(key[0], key[1])] = train_part[(train_part['Neighborhood'] == key[0]) 
                                    & (train_part['LotFrontage'].notnull())][['LotFrontage']].mean().round(2).to_dict()['LotFrontage']
    impute_index = full_set['LotFrontage'].isnull()
    full_set.loc[impute_index,'LotFrontage'] = full_set[impute_index].apply(lambda x:
                                        imputation_dict['LotFrontage'][(x['Neighborhood'], x['LotConfig'])], axis =1)
 
    nas = np.sum(full_set.isna()).reset_index()
    nas.columns = ['feature', 'NAs']
    nas.set_index('feature', inplace=True)
    nas = nas[nas['NAs']>0].sort_values('NAs',ascending=False)

    numerical_missing = list(set(nas.index) & (set(ordinal_columns) | set(numerical_columns)))

    # impute numerical or ordinal with median 
    for col in numerical_missing:
        if col in full_set.columns:
            med_val = full_set[col][1:1461].median()
            full_set[col].fillna(value = med_val, inplace = True)
        else:
            continue

    return full_set


def impute_cat_mode(full_set):
    '''
    Deal with actual missing values aka Na's
    '''
    nominal_columns = list(set(categorical_columns) - set(ordinal_columns))

    nas = np.sum(full_set.isna()).reset_index()
    nas.columns = ['feature', 'NAs']
    nas.set_index('feature', inplace=True)
    nas = nas[nas['NAs']>0].sort_values('NAs',ascending=False)

    nominal_missing = list(set(nas.index) & set(nominal_columns))

    # impute nominal features with mode
    for col in nominal_missing:
        if col in full_set.columns:
            mode_val = full_set[col][1:1461].mode()[0]
            full_set[col].fillna(value = mode_val, inplace = True)
        else:
            continue

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
    full_set = add_features(full_set)
    full_set = drop_features(full_set)
    
    return full_set


def add_features(full_set):
    # Add new feature

    full_set['propConsLot'] = full_set['1stFlrSF'] / full_set['LotArea']

    full_set['GarageArea'] = np.where((full_set['GarageArea'].isna()),
                                      full_set.loc[(full_set['GarageType']=='Detchd'),
                                                   'GarageArea'][1:1461].mean(),full_set['GarageArea'])

    full_set['garage'] = 0
    full_set['garage'] = np.where((full_set['GarageType'] == 'BuiltIn'),1,full_set['garage'])
    full_set['garage'] = np.where((full_set['GarageType'] == 'Basment'),1,full_set['garage'])
    full_set['garage'] = np.where((full_set['GarageType'] == '2Types'),0.5,full_set['garage'])

    
    full_set['livingArea'] = (full_set['TotalBsmtSF'] + full_set['GrLivArea'] 
                              + full_set['GarageArea'] * full_set['garage'])

    full_set['constArea'] = full_set['1stFlrSF'] + full_set['GarageArea'] * full_set['garage']

    full_set['yard'] = 0
    full_set['yard'] = full_set['LotArea'] - full_set['constArea']  #!!!!!!!!!!!!! should we keep this??????????

    full_set['constCost'] = full_set['GrLivArea'] * 100 + 4 * full_set['LotArea'] #!!and this???
        
    full_set['yardPerc'] = full_set['constArea'] / full_set['LotArea']

    full_set['bathrooms'] = full_set['FullBath'] + full_set['HalfBath']
    
    temp_dict = dict([['StoneBr1Fam',90.4395973747988],['StoneBrTwnhsE',64.9650413081317],['NridgHt1Fam',80.1543151289586],
                      ['NridgHtTwnhs',65.1602983195312],['NridgHtTwnhsE',69.1353791634465],['NoRidge1Fam',72.0738507675531],
                      ['Veenker1Fam',66.0429078276295],['VeenkerTwnhsE',80.5042194092827],['Gilbert1Fam',68.3854903542442],
                      ['Gilbert2fmCon',54.0234724292101],['Somerst1Fam',68.2144072973843],['SomerstTwnhs',63.2701103604549],
                      ['SomerstTwnhsE',67.1788824758477],['Timber1Fam',66.7389414909238],['Timber2fmCon',60.0288411116938],
                      ['TimberTwnhsE',68.7389414909238],['Crawfor1Fam',66.3725911426128],['Crawfor2fmCon',54.5893602225313],
                      ['CrawforDuplex',42.227593738624],['CrawforTwnhsE',80.6373302358828],['CollgCr1Fam',62.6262950167203],
                      ['CollgCrDuplex',47.6262950167203],['CollgCrTwnhsE',67.8023156899811],['ClearCr1Fam',61.2119033661411],
                      ['ClearCrTwnhs',58.2119033661411],['SawyerW1Fam',61.1450936373841],['SawyerWDuplex',55.8753295978906],
                      ['SawyerWTwnhsE',54.9017441230636],['Blmngtn1Fam',54.7578767123288],['BlmngtnTwnhsE',60.150351985043],
                      ['NWAmes1Fam',56.301293015552],['NWAmes2fmCon',43.301293015552],['NWAmesDuplex',35.2847132104853],
                      ['BluesteTwnhs',55.431718061674],['BluesteTwnhsE',55.0603907637655],['Mitchel1Fam',57.6274194214023],
                      ['Mitchel2fmCon',47.3372781065089],['MitchelDuplex',46.4664009742057],['MitchelTwnhs',57.8687367678193],
                      ['MitchelTwnhsE',55.5553243149497],['BrkSide1Fam',55.1598006041819],['BrkSide2fmCon',51.792828685259],
                      ['NPkVillTwnhs',53.7742170429716],['NPkVillTwnhsE',54.6368773137624],['Sawyer1Fam',54.9031233267892],
                      ['Sawyer2fmCon',50.9279628195197],['SawyerDuplex',41.0229742817174],['NAmes1Fam',53.9106785412273],
                      ['NAmes2fmCon',71.9039548022599],['NAmesDuplex',42.0147234693277],['NAmesTwnhsE',61.4473581213307],
                      ['BrDaleTwnhs',53.5906850987746],['BrDaleTwnhsE',47.9123945489942],['Edwards1Fam',50.1959140532244],
                      ['Edwards2fmCon',49.3992365555181],['EdwardsDuplex',56.7003610108303],['EdwardsTwnhs',43.3146997929607],
                      ['EdwardsTwnhsE',71.0762529983588],['MeadowVTwnhs',57.2698772426818],['MeadowVTwnhsE',47.9567654841458],
                      ['SWISU1Fam',49.4898887276246],['SWISU2fmCon',49.8638511814982],['SWISUDuplex',41.6768699545614],
                      ['OldTown1Fam',49.2350787555647],['OldTown2fmCon',41.1327129667974],['OldTownDuplex',47.7662535079514],
                      ['IDOTRR1Fam',45.1213106618607],['IDOTRR2fmCon',58.958157227388],['IDOTRRDuplex',51.2104283054004]])
    
    full_set['neigh_BldgType'] = list(map(lambda x,y: temp_dict[str(x) + str(y)],
                                          full_set['Neighborhood'],full_set['BldgType']))

    # Combine 3SsnPorch, EnclosedPorch, ScreenPorch, OpenPorchSF, and WoodDeckSF into 1 ordinal feature ranking the types
    if all( x in full_set.columns for x in ['3SsnPorch','EnclosedPorch','ScreenPorch','OpenPorchSF','WoodDeckSF']):

        asdf = pd.DataFrame(5 if full_set['3SsnPorch'].loc[i] > 0 else 
                                         4 if full_set['EnclosedPorch'].loc[i] > 0 else 
                                         3 if full_set['ScreenPorch'].loc[i] > 0 else 
                                         2 if full_set['OpenPorchSF'].loc[i] > 0 else 
                                         1 if full_set['WoodDeckSF'].loc[i]> 0 else 0 
                                         for i in range(1,1+len(full_set[['WoodDeckSF']])))
        asdf['Id']=asdf.index+1
        full_set['porchType'] = asdf.set_index('Id')
        full_set['porchArea'] = (full_set['3SsnPorch'] + full_set['EnclosedPorch'] + full_set['ScreenPorch'] + 
            full_set['OpenPorchSF'] + full_set['WoodDeckSF'])
        full_set.drop(columns = ['3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'OpenPorchSF', 'WoodDeckSF'], inplace = True)

    # Combine PoolArea and PoolQC to PoolYN
    if 'PoolArea' in full_set.columns:
        asdf = pd.DataFrame(1 if full_set['PoolArea'].loc[i] > 0 else 0 for i in range(1,1+len(full_set['PoolArea'])))
        asdf['Id']=asdf.index+1
        full_set['poolYN'] = asdf.set_index('Id')
        full_set.drop(columns = 'PoolArea', inplace = True)
        if 'PoolQC' in full_set.columns:
            full_set.drop(columns = 'PoolQC', inplace = True)

    # Convert Fence to FenceYN
    if 'Fence' in full_set.columns:
        asdf = pd.DataFrame(1 if full_set['Fence'].loc[i] != 'No' else 0 for i in range(1,1+len(full_set['Fence'])))
        asdf['Id']=asdf.index+1
        full_set['fenceYN'] = asdf.set_index('Id')
        full_set.drop(columns = 'Fence', inplace = True)

    full_set['roomForCar'] = full_set['GarageArea']/full_set['LotFrontage']
    
    full_set['curbAppeal'] = [x*y if y>0 else x for x, y in zip(full_set['LotFrontage'], full_set['MasVnrArea'])]
    
    full_set['bsmtnicety'] = (full_set['BsmtExposure'] + 1)*(full_set['BsmtQual'] + 1)*(full_set['TotalBsmtSF'])
    
    return full_set


def drop_features(full_set):
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


def keep_these_c(full_set,colnames_):
#     list(set(colnames_)|set('SalePrice'))
    full_set = full_set[list(set(colnames_)|set(['SalePrice']))]
    
    return full_set


def drop_these_c(full_set,colnames_):
    for x in colnames_:
        if x in full_set.columns:
            full_set.drop(columns = x, inplace = True)
        else:
            print("Couldn't drop {}",x)
    
    return full_set


def find_cooks(full_set):
    '''
    Takes numerical features to find highly leveraged outliers
    Returns indices of outliers as list
    '''
    import statsmodels.api as sm
    outliers = []
    check_these = list(set(full_set.columns))
    check_these.remove('LotOnRoad')
    for x in check_these:
        if x in nominal_columns:
            pass
        elif full_set[x].isna().sum()>0:
            pass
        else:
            model = sm.OLS(full_set['SalePrice'],full_set[x])
            results = model.fit()
            influence = results.get_influence()
            cooks_dis = influence.summary_frame()
            outliers += cooks_dis.index[cooks_dis['cooks_d']> 0.5 ].tolist()
    outliers = list(set(outliers))
    
    return outliers

def var_inf_fac(data):
    '''
    # CHECK FOR MULTICOLLINEARITY USING VIF  for multicollinearity using VIF =============================
    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    X = add_constant(data) # intercept
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],index=X.columns)
    return vif.sort_values(ascending = False)

def drop_these_r(data):
    '''
    Drop outliers with high leverage
    '''
    data= data.drop([31,411,524,534,592,677,689,804,1047])
    data= data.drop([363,390,496,633,729,770,899,969,1063,1182])
    data= data.drop([250, 314, 336, 707, 1299])
    
    return data


def label_encode(full_set):
    '''
    Label encoding nominal features
    '''
    categorical_columns = []

    for col in full_set.columns:
        if full_set[col].dtypes in ['object']:
            categorical_columns.append(col)

    nominal_columns = list(set(categorical_columns)-set(ordinal_columns))
    nominal_kept = set(nominal_columns) & set(full_set.columns)
    full_set = pd.get_dummies(full_set, prefix_sep = '_', columns=nominal_kept, drop_first = True)

    return full_set    


def transform_f(data):
    '''
    Manual transformation
    '''
    columns_to_change = ['LotArea','TotalBsmtSF','GrLivArea']
    #SQUARE OF VARIABLES
    these_columns = list(set(data.columns)&set(columns_to_change))
    for col in these_columns:
        new_col = str(col) + "_sq"
        data[new_col] = np.log(data[col]+1)

    return data


def find_boxcox(x,y):
    from scipy.stats import boxcox
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    xTrbc = x.copy()
    lm.fit(xTrbc, y)
    default_s = lm.score(xTrbc, y)
    print('no transformation: ',lm.score(xTrbc, y))

    to_boxcox = []
    for i in list(set(x.columns) & set(numerical_columns))[:]:
        xTrbc = x.copy()
        xTrbc[[i]] = boxcox(x[[i]] + 1)[0]
        lm.fit(xTrbc, y)
        if lm.score(xTrbc, y) > default_s:
            to_boxcox.append(i)
            print(i, ': ', lm.score(xTrbc, y) - default_s)

    return to_boxcox


def boxcox_these(full_set):
    '''
    Takes non-ordinal numerical features to apply boxcox transformation
    '''
    from scipy.stats import boxcox
    boxcoxing = ['ConstArea', 'LivingArea', 'YardPerc', 'LotArea', '2ndFlrSF', 'BsmtUnfSF', 'BsmtFinSF1', 
                 'LowQualFinSF', '1stFlrSF', 'MiscVal', 'BsmtFinSF2', 'Yard', 'ConstCost', 'PropConsLot']
    boxcoxing = list(set(boxcoxing) & set(full_set.columns))
    for i in boxcoxing:
        full_set[[i]] = boxcox(full_set[[i]] + 1)[0]

    return full_set


def split_xy(full_set):
    
    yTr = full_set['SalePrice'][full_set['SalePrice']>=0]
    xTr = full_set[full_set['SalePrice']>=0].drop(['SalePrice'], axis=1)
    xTe = full_set[full_set['SalePrice']==-1].drop(['SalePrice'], axis=1)
    
    return xTr,yTr,xTe


def bc_price(yTr):
    '''
    Transform the Price
    '''
    from scipy.stats import boxcox
    yTr, transf_yTr_lambda = boxcox(yTr)[:2]
    
    return yTr, transf_yTr_lambda


def log_price(yTr):
    '''
    Transform the Price w/ logs
    '''
    import numpy as np
    yTr = np.log(yTr)
    lamd = 0
    return yTr, lamd


def graph_lin_resi_v_y(xTr,yTr):
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(xTr,yTr)
    plt.scatter(x = lm.predict(xTr),y = yTr - lm.predict(xTr), s= 0.2)
    print(lm.score(xTr,yTr))
    print(xTr.shape)


def comp_bxcxed_graph(xTr,yTr):
    import matplotlib.pyplot as plt
    from scipy.stats import boxcox

    non_nominal = pd.Series(list(set(ordinal_columns) | set(numerical_columns)), name = 'Feature')

    lm = LinearRegression()
    relevance = pd.DataFrame(columns = ['Feature','Rsq'])
    for colnum in range(0,len(xTr.columns)):
        col = xTr.columns[colnum]
        lm.fit(xTr[[col]],yTr)
        relevance.loc[colnum,'Feature'] = col
        relevance.loc[colnum,'Rsq'] = lm.score(xTr[[col]],yTr)
        
    residual = pd.DataFrame(columns = relevance['Feature'])
    these_resids = pd.merge(relevance['Feature'],non_nominal,how = 'inner')

    to_boxcox = ['constArea', 'livingArea', 'yardPerc', 'LotArea', '2ndFlrSF', 'BsmtUnfSF', 'BsmtFinSF1', 
                 'LowQualFinSF', '1stFlrSF', 'MiscVal', 'BsmtFinSF2', 'yard', 'constCost', 'propConsLot']
    to_boxcox = list(set(to_boxcox)&set(x.columns))
    leftright = 2
    updownmax = 25
    updown = updownmax if len(to_boxcox) > updownmax else len(to_boxcox)//leftright

    fig, ax = plt.subplots(updown, leftright,squeeze = False,figsize = (15,updown * 5))
    for i in range(updown):
        for j in range(leftright):
            if j%2 ==0:
                lm.fit(xTr[[to_boxcox[i]]],yTr)
                ax[i, j].scatter(x = lm.predict(xTr[[to_boxcox[i]]]),
                                 y = yTr - lm.predict(xTr[[to_boxcox[i]]]))
                ax[i, j].title.set_text(to_boxcox[i])
            else:
                xTrbc = pd.DataFrame(boxcox(xTr[[to_boxcox[i]]] + 1)[0])
                lm.fit(xTrbc,yTr)
                ax[i, j].scatter(x = lm.predict(xTrbc),
                                 y = yTr - lm.predict(xTrbc))
    fig
    
    
def evaluate_model(model, x, y, folds, lmbda=None):
    from scipy.special import inv_boxcox
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_log_error
    print("Accuracy: ",round(model.score(x, y),4))
    if lmbda == None:
        ypred = cross_val_predict(model, x, y, cv = folds, n_jobs = -1)
        print("RMSLE = ", round(np.sqrt(mean_squared_log_error(ypred, y )),4) )
        scores   = -cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv = folds)
        print("Cross-Val Score =",round(np.mean(scores**.5)))
    else:
        ypred = cross_val_predict(model, x, y, cv = folds, n_jobs = -1)
        ypred = inv_boxcox(ypred, lmbda)
        #ypred = ((ypred*lmbda)+1)**(1/lmbda)
        #y = ((y*lmbda)+1)**(1/lmbda)
        y = inv_boxcox(y, lmbda)
        print("RMSLE = ", round(np.sqrt(mean_squared_log_error(ypred, y )),4) )
        scores = -cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv = folds)
        print("Cross-Val Score =",round(np.mean(scores**.5)))

        
def o_l_s(x,y,cv = 5, transf_yTr_lambda=None):
    from sklearn import linear_model
    np.random.seed(0)
    print("\nLinear","- "*20)
    ols = linear_model.LinearRegression()
    ols.fit(x, y)
    print(np.exp(ols.intercept_))
    print(list(zip(np.exp(ols.coef_),x.columns)))
    evaluate_model(ols, x, y, 5, transf_yTr_lambda)
    
    
def ols_pred(x,y,xTe,transf_yTr_lambda=None):
    from sklearn import linear_model
    ols = linear_model.LinearRegression()
    ols.fit(x, y)
    temp = pd.DataFrame(ols.predict(xTe))
    if transf_yTr_lambda == 0:
        temp = np.exp(temp)
    elif transf_yTr_lambda == None: 
        temp = temp
    else:
        temp = ((temp*transf_yTr_lambda)+1)**(1/transf_yTr_lambda)
    idx = range(1461,2920)
    idx = xTe.index
    temp.index=idx
    temp=temp.reset_index()
    temp.columns=['Id','SalePrice']
    temp.to_csv('olspl.csv',index=False)
    
    
def lasso_pred(x,y,xTe,transf_yTr_lambda=None):
    from sklearn import linear_model
    lasso = linear_model.LassoCV(eps=0.001, n_alphas=100, cv=5, normalize=True, max_iter=1000000)
    lasso.fit(x, y)
    temp = pd.DataFrame(lasso.predict(xTe))
    if transf_yTr_lambda == 0:
        temp = np.exp(temp)
    elif transf_yTr_lambda == None: 
        temp = temp
    else:
        temp = ((temp*transf_yTr_lambda)+1)**(1/transf_yTr_lambda)
    idx = range(1461,2920)
    temp.index=idx
    temp=temp.reset_index()
    temp.columns=['Id','SalePrice']
    temp.to_csv('lassopl.csv',index=False)
    
    
def ridge(x,y,cv = 5, transf_yTr_lambda=None):
    from sklearn import linear_model    
    np.random.seed(1)
    print("\nRidge","- "*20)
    ridge = linear_model.RidgeCV(alphas=np.arange(0.05,4,0.01),normalize=[True,False],fit_intercept=[True,False])
    ridge.fit(x, y)
    evaluate_model(ridge, x, y, 5, transf_yTr_lambda)
    
    ridge_betas = pd.DataFrame(ridge.coef_, x.columns)
    ridge_betas = ridge_betas.reset_index()
    ridge_betas.columns = ['feature','beta']
    ridge_betas['abs_beta'] = abs(ridge_betas['beta'])
    ridge_betas = ridge_betas.sort_values('abs_beta',ascending=False)
    ridge_betas.to_csv('ridge_betas.csv')


def lasso(x,y,cv = 5, transf_yTr_lambda=None):
    from sklearn import linear_model
    print("\nLasso","- "*20)
    lasso = linear_model.LassoCV(eps=0.001, n_alphas=100, cv=5, normalize=True, max_iter=1000000)
    lasso.fit(x, y)
    print("Best alpha = ",lasso.alpha_)
    evaluate_model(lasso, x, y, 5, transf_yTr_lambda)
    print(list(zip(np.exp(lasso.coef_),x.columns)))
    
    lasso_betas = pd.DataFrame(lasso.coef_, x.columns)
    lasso_betas = lasso_betas.reset_index()
    lasso_betas.columns = ['feature','beta']
    lasso_betas['abs_beta'] = abs(lasso_betas['beta'])
    lasso_betas = lasso_betas.sort_values('abs_beta',ascending=False)
    lasso_betas.to_csv('lasso_betas.csv')
    

def rank_feat(filename):
    import re
    beta = pd.read_csv(filename)
    rank = [re.split('_',x)[0] if re.search('_',x) else x for x in list(beta['feature'])]
    imp = list(set(rank))
    imp = pd.DataFrame(imp, columns = ['Feature'])
    imp['Rank'] = [rank.index(x) for x in imp['Feature']]
    imp.sort_values('Rank',inplace = True)

    imp.reset_index(inplace = True)
    imp.drop(columns = ['Rank','index'],inplace = True)
    return imp[:25]

def add_rank(to_here,from_here,colname):
    for x in to_here['Feature']:
        if x in from_here['Feature'].unique():
            to_here.loc[to_here['Feature'] == x,colname] = from_here.index[from_here['Feature'] == x].tolist()[0]
        else:
            to_here.loc[to_here['Feature'] == x,colname] = 30
    return to_here

def rank_feats():
    from sklearn.linear_model import LinearRegression
    import re

    lm = LinearRegression()
    relevance = pd.DataFrame(columns = ['Feature','Rsq'])
    for colnum in range(0,len(x.columns)):
        col = x.columns[colnum]
        lm.fit(x[[col]],y)
        relevance.loc[colnum,'Feature'] = col
        relevance.loc[colnum,'Rsq'] = lm.score(x[[col]],y)

    relevance.sort_values('Rsq',ascending = False, inplace = True)
    rank = list(set([re.split('_', i)[0] if re.search('[_]',i) else i for i in list(relevance['Feature'])]))
    lin_imp = list(set(rank))
    lin_imp = pd.DataFrame(lin_imp, columns = ['Feature'])
    lin_imp['Rank'] = [rank.index(i) for i in lin_imp['Feature']]
    lin_imp.sort_values('Rank',inplace = True)
    lin_imp.reset_index(inplace = True)
    lin_imp.drop(columns = ['Rank','index'], inplace = True)
    lin_imp = lin_imp[:25]
    
    
    ridge_beta = rank_feat('ridge_betas.csv')
    lasso_beta = rank_feat('lasso_betas.csv')
#     boost_beta = rank_feat('boostFeatImp.csv')
#     forest_beta = rank_feat('RandForestFeatImp.csv')
    all_feats = pd.DataFrame(set(ridge_beta['Feature'])|set(lasso_beta['Feature'])|set(lin_imp['Feature'])
#                              |set(boost_beta['Feature'])|set(forest_beta['Feature'])
                             , columns = ['Feature'])
    all_feats = add_rank(all_feats,lin_imp,'lin')
    all_feats = add_rank(all_feats,ridge_beta,'ridge')
    all_feats = add_rank(all_feats,lasso_beta,'lasso')
#     all_feats = add_rank(all_feats,boost_beta,'boost')
#     all_feats = add_rank(all_feats,forest_beta,'forest')
    all_feats['Sum'] = all_feats['lin']+all_feats['ridge']+all_feats['lasso']#+all_feats['boost']+all_feats['forest']
    all_feats.sort_values('Sum',inplace = True)
    all_feats.reset_index(inplace = True)
    all_feats.drop(columns = ['index'],inplace = True)
    print(all_feats)
    common_feats = pd.DataFrame(set(ridge_beta['Feature'])&set(lasso_beta['Feature'])&set(lin_imp['Feature'])
#                              &set(boost_beta['Feature'])&set(forest_beta['Feature'])
                                , columns = ['Feature'])


#RUN THIS#####################################


import pandas as pd
import numpy as np
train = pd.read_csv('train.csv')
train = train.set_index('Id')    
test = pd.read_csv('test.csv')
test = test.set_index('Id')
    
categorical_columns = ['MSSubClass','MSZoning','MasVnrType','PoolQC','MiscFeature','Street','Alley','LotShape',
                       'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
                       'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
                       'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                       'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                       'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','Fence',
                       'SaleType','SaleCondition','MoSold','YrSold']

numerical_columns = ['LotArea','OverallQual','MasVnrArea','PoolArea','OverallCond','YearBuilt','YearRemodAdd',
                     'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
                     'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                     'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
                     'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','LotFrontage']

ordinal_columns = ['ExterQual','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageFinish','BsmtExposure']

nominal_columns = list(set(categorical_columns)-set(ordinal_columns))

lmda = None

#Putting training and testing dataset together to evaluate NA's and Engineer Features
test['SalePrice'] = -1
full_set = pd.concat([train,test],axis=0)
full_set = full_set[train.columns]



full_set = deal_with_NAs(full_set)
full_set = add_features(full_set)
full_set = keep_these_c(full_set,best3 + ['curbAppeal'])
full_set = drop_these_r(full_set)
full_set = label_encode(full_set)
# full_set = transform_f(full_set)
# full_set = boxcox_these(full_set)
x, y, xte = split_xy(full_set)
# vif = var_inf_fac(full_set)
# comp_bxcxed_graph(x,y)
# y, lmda = bc_price(y)
y, lmda = log_price(y)

graph_lin_resi_v_y(x,y)
lasso(x,y,5,lmda)
ridge(x,y,5,lmda)
o_l_s(x,y,5,lmda)
ols_pred(x,y,x,lmda)
lasso_pred(x,y,xte,lmda)

