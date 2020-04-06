import seaborn as sns
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import glob, os, sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import altair as alt

#set up pandas view 
alt.renderers.enable('altair_viewer')
pd.set_option('display.max_columns', None)



# dataPath= r"D:\School\Deep-Learning\assignment1\house-prices-advanced-regression-techniques"
# trainData= dataPath+r"\train.csv"
###===========define data path ================###

def data():
    rootPath=os.getcwd()
    trainData=os.path.join(rootPath,"train.csv")
    df_train=pd.read_csv(trainData)
    testData=os.path.join(rootPath,"test.csv")
    df_test=pd.read_csv(testData)

    print(df_train.describe(include=['O'])) # found Alley/PoolQC/MiscFeature/Fence missing values


    train_corr =df_train.corr() #計算相關係數
    high_corr=train_corr.index[abs(train_corr["SalePrice"]>0.3)]

    float_data = train_corr.index #是float或int的
    all_col = df_train.columns #全部的col
    object_data = []
    for i in range(len(all_col)): #查找全部的all_col，len(all_col)是長度(要全部找過一遍)
        if all_col[i] not in float_data: #如果在float_data裡面沒有，表示它是object幫的
            object_data.append(all_col[i]) #不是就加上去


    for i in object_data:
    	df_train[i]= LabelEncoder().fit_transform(df_train[i].factorize()[0])

    print(df_train)


    for i in df_train.columns: #查找原本資料中所有columns
        if i not in high_corr: #如果沒有相關係數大於0.3的話
            df_train = df_train.drop(i,axis=1) #就把它拔掉
    print(df_train)

    for col in df_train.columns:
        if df_train[col].isnull().values.any():
            df_train[col].fillna(0, inplace=True)

    train_targets = df_train["SalePrice"].values #把SalePrice這行數值整個拉出來
    train_data = df_train.drop(columns=["SalePrice"]) #刪除SalePrice這行
    print(train_targets)
    print("*"*50)
    print(train_data)
    print(train_data.describe())
    # plt.subplots(figsize=(80, 80))  # 設置長寬尺寸大小
    # sns.heatmap(trian_corr, annot=True, vmax=1, cmap="YlGnBu")
    # plt.savefig('test.png')


    X_train,X_validation,Y_trian,Y_val = train_test_split(train_data, train_targets, test_size=0.2, random_state=0)



    print(X_train.shape)
    print(Y_trian.shape)
    print('*'*50)
    print(X_validation.shape)
    print(Y_val.shape)
    X_train_dataset = X_train.values #取出數值，轉換回list
    X_validation_dataset = X_validation.values
    print(X_train_dataset)

    normalize = preprocessing.StandardScaler() 

    X_trian_normal_data = normalize.fit_transform(X_train_dataset) #將訓練資料標準化
    X_val = normalize.fit_transform(X_validation_dataset) #將驗證資料標準化

    print(X_trian_normal_data)
    print('*'*50)
    print(X_val)




    return X_trian_normal_data,Y_trian,X_val,Y_val


#Create Model
def create_model(X_train,Y_train,X_val,Y_val):
    from keras.models import Sequential
    from keras import layers
    from keras import optimizers
    from keras.layers import BatchNormalization,Dropout,Dense
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.models import load_model


    model = Sequential()
    model.add(layers.Dense(1024,kernel_initializer='random_normal',activation='relu',input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))#Dropout: a simple way to prevent neural networks from overfitting
    model.add(layers.Dense(1024,kernel_initializer='random_normal',activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(512,kernel_initializer='random_normal',activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(256,kernel_initializer='random_normal',activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(128,kernel_initializer='random_normal',activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(32,kernel_initializer='random_normal',activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(16,kernel_initializer='random_normal',activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(1,kernel_initializer='random_normal',activation='linear'))
    adam=optimizers.Adam(lr=0.001)

    model.compile(optimizer=adam,loss='mae')
    
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)
    ckptCallback = ModelCheckpoint('latest.h5',monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=True,mode='auto',period=1)
    history=model.fit(X_train,Y_trian,validation_data=[X_val,Y_val],callbacks=[early_stopping_callback,ckptCallback],
    epochs=300,batch_size=256,verbose=2)

    return model,history

X_trian_normal_data,Y_trian,X_validation_normal_data,Y_validation=data()
model,history = create_model(X_trian_normal_data,Y_trian,X_validation_normal_data,Y_validation)    


# best_run, best_model = optim.minimize(model=create_model,
#                                       data=data,
#                                       algo=tpe.suggest,
#                                       max_evals=3,
#                                       trials=Trials())
                                      

    

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.savefig('best.png')

# # history 會記錄訓練的狀況，會將model.fit回傳的東西記錄下來，以下是常用的示範
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc = 'upper right')
# plt.savefig('nice_mae_18_columns.h5.png')




# #remove addtional columns 
# for i in df_test.columns:
#     if i not in high_corr:
#         df_test=df_test.drop(i,axis=1)
# #remove null value        
# for col in df_test.columns:
#     if df_test[col].isnull().values.any():
#         df_test[col].fillna(0, inplace=True)


# X_test_dataset = df_test.values
# X_test_normal_data = normalize.fit_transform(X_test_dataset)
# print(X_test_normal_data) #看一下標準化的結果


# model.load_weights('nice_mae_18_columns.h5')
# pred = model.predict(X_test_normal_data)
# np.set_printoptions(threshold=sys.maxsize)
# print("This is prediction of 2nd round : \n",pred)

# with open('house_predict_mse_18_columns.csv', 'w') as f:
#     f.write('id,SalePrice\n')
#     for i in range (len(pred)):
#         f.write(str(i+1461)+','+str(float(pred[i]))+'\n')







