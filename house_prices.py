import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn .model_selection import train_test_split


train = pd.read_csv('train.csv')
train = train.set_index('Id')
train.head()
train.plot(kind='scatter', x='MSSubClass',y='SalePrice')

#train and test splits
Tr,Te = train_test_split(train, test_size=0.2, random_state=1)
yTr =  Tr['SalePrice'] 
yTe =  Te['SalePrice']
xTr = Tr.drop('SalePrice',axis=1)
xTe = Te.drop('SalePrice',axis=1)

