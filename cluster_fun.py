
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def cluster(n, df): 
    '''
    ARGS:   N = NUMBER OF CLUSTERS FOR KMEANS, 
            DF = DATAFRAME OF 1 CATEGORICAL VARIABLE WITH SALES PRICE
            TARGETVAR = VARIABLE TO CLUSTER (E.G. NEIGHBORHOODS ETC..)


    RETURNS DATAFRAME WITH (ORDERED) LABELS ASSIGNED TO CATEGORICAL COLUMN
    '''      
    xvar = df.columns[0]
    yvar = df.columns[1]

    #MAKE LABELS OF CATEGORICAL VARIABLE BECAUSE K MEANS CAN'T HANDLE STRINGS


    df_means_og = df.groupby(xvar).agg("mean").reset_index().sort_values(yvar).reset_index()
    df_x = df_means_og[xvar]
    df_y = df_means_og[yvar]


    ary = np.array(df_y)

    #APPLY K MEANS
    kfit = KMeans(n_clusters=n, random_state=0).fit(ary.reshape(-1,1))
    klabels = kfit.labels_
    klab_df = pd.DataFrame(klabels)

    ary_df = pd.DataFrame(df_y)
    ary_df['klab'] = klabels
    ary_df[xvar] = df_x

    u, ind = np.unique(klabels,return_index=True)
    u = u[np.argsort(ind)]
    klabels_ordered = list(range(1,len(u)+1))
    kordered_dict = dict(zip(u, klabels_ordered))
    
    
    k_df = pd.DataFrame(df_x)
    k_df['klabels'] = klabels
    k_df['klabels'] = pd.Series(k_df.klabels.map(kordered_dict))
    colnames = df.columns
    
    k_df.columns = [xvar, 'klabels']

    kDict = k_df

    return kDict


#Example:
#data = pd.read_csv('data_clean.csv')
#train = data[0:1460]

#cluster(2, train[['MSZoning', 'SalePrice']])

#test = cluster(5,train[['Neighborhood', 'SalePrice']])




def kReplace(df, kdf, xvar, yvar): 

    '''
    Replace original column with klabeled column in data 

    ARGS:   DF: ORIGINAL DATA
            KDF: OUTPUT OF CLUSTER FUNCTION
            XVAR: E.G. NEIGHBORHOOD
            YVAR: E.G. SALE PRICE
    '''

    kData = df
    ktargetname = 'k' + xvar
    kData[ktargetname]= ""
    
    #kdf = kdf.drop(yvar, axis = 1)
    kdf = kdf.set_index(xvar).transpose().to_dict(orient='list')
   
    for k, v in kdf.items():
        kdf[k] = v[0]

    for i in np.arange(len(df)):
        kData[ktargetname].loc[i]  = kdf.get(df[xvar].loc[i])
        
    return kData

