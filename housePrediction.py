import seaborn as sns
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



dataPath= r"D:\School\Deep-Learning\assignment1\house-prices-advanced-regression-techniques"
trainData= dataPath+r"\train.csv"
df_train=pd.read_csv(trainData)

train_corr =df_train.corr() #計算相關係數

high_corr=train_corr.index[abs(train_corr["SalePrice"]>0.5)]

float_data = train_corr.index #是float或int的的傢伙
all_col = df_train.columns #全部的col
object_data = []
for i in range(len(all_col)): #查找全部的all_col，len(all_col)是長度(要全部找過一遍)
    if all_col[i] not in float_data: #如果在float_data裡面沒有，表示它是object幫的
        object_data.append(all_col[i]) #不是就加上去


for i in object_data:
	df_train[i]= LabelEncoder().fit_transform(df_train[i].factorize()[0])
print(df_train)

trian_corr =df_train.corr()
high_corr = trian_corr.index[abs(trian_corr["SalePrice"])>0.6]
print(high_corr)
# plt.subplots(figsize=(30, 10))  # 設置長寬尺寸大小
# sns.heatmap(trian_corr, annot=True, vmax=1, cmap="YlGnBu")
# plt.show()