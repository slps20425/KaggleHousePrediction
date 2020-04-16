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
from keras.callbacks import ModelCheckpoint, EarlyStopping

#set up pandas view 
alt.renderers.enable('altair_viewer')
pd.set_option('display.max_columns', None)

rootPath=os.getcwd()
trainData=os.path.join(rootPath,"train.csv")
df_train=pd.read_csv(trainData)
testData=os.path.join(rootPath,"test.csv")
df_test=pd.read_csv(testData)
normalize = preprocessing.StandardScaler() 


# dataPath= r"D:\School\Deep-Learning\assignment1\house-prices-advanced-regression-techniques"
# trainData= dataPath+r"\train.csv"
###===========define data path ================###

def data():
    global trainData,df_train,df_test,normalize
    print(df_train.describe(include=['O'])) # found Alley/PoolQC/MiscFeature/Fence missing values
    df_train=df_train.drop(['Alley','PoolQC','MiscFeature','Fence'],axis=1)
    df_test=df_test.drop(['Alley','PoolQC','MiscFeature','Fence'],axis=1)

    categorical = [var for var in df_train.columns if df_train[var].dtype=='O']
    numerical = [var for var in df_train.columns if df_train[var].dtype!='O']
    discrete = []   
    for var in numerical:
        if len(df_train[var].unique())<15:
            discrete.append(var)
    continuous = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice']]

    #處理nan value, categorical >> replace with None ; numerical >> mean or median
    #處理 nan in numerical discrete/continuous and categorical
    for col in discrete:
        if df_train[col].isnull().mean()>0:
            df_train[col].fillna(df_train[col].median(), inplace=True)
        if df_test[col].isnull().mean()>0:
            df_test[col].fillna(df_train[col].median(), inplace=True)
    for col in continuous:
        if df_train[col].isnull().mean()>0:
            df_train[col].fillna(df_train[col].mean(), inplace=True)
        if df_test[col].isnull().mean()>0:
            df_test[col].fillna(df_train[col].mean(), inplace=True)

    for col in categorical:
        if df_train[col].isnull().mean()>0:
            df_train[col].fillna("None", inplace=True)
        if df_test[col].isnull().mean()>0:
            df_test[col].fillna("None", inplace=True)

    for col in categorical:
        df_train[col]= LabelEncoder().fit_transform(df_train[col].factorize()[0])
        df_test[col]= LabelEncoder().fit_transform(df_test[col].factorize()[0])

    train_corr =df_train.corr() #計算相關係數
    high_corr=train_corr.index[abs(train_corr["SalePrice"]>0.35)]

    for i in df_train.columns: #查找原本資料中所有columns
        if i not in high_corr: #如果沒有相關係數大於0.3的話
            df_train = df_train.drop(i,axis=1) #就把它拔掉
            df_test = df_test.drop(i,axis=1)

    train_targets = df_train["SalePrice"].values #把SalePrice這行數值整個拉出來
    train_data = df_train.drop(columns=["SalePrice"]) #刪除SalePrice這行
    # print(train_targets)
    # print("*"*50)
    # print(train_data)
    # print(train_data.describe())
    # plt.subplots(figsize=(80, 80))  # 設置長寬尺寸大小
    # sns.heatmap(trian_corr, annot=True, vmax=1, cmap="YlGnBu")
    # plt.savefig('test.png')

    #分data 80 train data ; 20 validation data
    X_train,X_temp,Y_trian,Y_temp = train_test_split(train_data, train_targets, test_size=0.2, random_state=0)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp,Y_temp,test_size=0.5,random_state=0)


    # print(X_train.shape)
    # print(Y_trian.shape)
    # print('*'*50)
    # print(X_validation.shape)
    # print(Y_val.shape)
    X_train_dataset = X_train.values #取出數值，轉換回list
    X_validation_dataset = X_val.values
    X_test=X_test.values
    print(X_train_dataset)

    

    X_trian_normal_data = normalize.fit_transform(X_train_dataset) #將訓練資料標準化
    X_val = normalize.fit_transform(X_validation_dataset) #將驗證資料標準化
    X_test=normalize.fit_transform(X_test)

    print(X_trian_normal_data)
    print('*'*50)
    print(X_val)


    X_test_dataset = df_test.values
    print(X_test_dataset)
    predictData = normalize.fit_transform(X_test_dataset)
    print('predict_test: \n',predictData) #看一下標準化的結果


    return X_trian_normal_data,Y_trian,X_val,Y_val,X_test,Y_test,predictData


#Create Model
def create_model():
    from keras.models import Sequential
    from keras import layers
    from keras import optimizers
    from keras.layers import BatchNormalization,Dropout,Dense
    from keras.models import load_model


    model = Sequential()
    model.add(layers.Dense(1024,kernel_initializer='random_normal',activation='selu',input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))#Dropout: a simple way to prevent neural networks from overfitting
    model.add(layers.Dense(1024,kernel_initializer='random_normal',activation='selu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(512,kernel_initializer='random_normal',activation='selu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(512,kernel_initializer='random_normal',activation='selu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(512,kernel_initializer='random_normal',activation='selu'))
    model.add(Dropout(0.15))
    model.add(layers.Dense(512,kernel_initializer='random_normal',activation='selu'))
    model.add(Dropout(0.15))
    model.add(layers.Dense(256,kernel_initializer='random_normal',activation='selu'))
    model.add(Dropout(0.15))
    model.add(layers.Dense(128,kernel_initializer='random_normal',activation='selu'))
    model.add(Dropout(0.15))
    model.add(layers.Dense(32,kernel_initializer='random_normal',activation='selu'))
    model.add(Dropout(0.15))
    model.add(layers.Dense(16,kernel_initializer='random_normal',activation='selu'))
    model.add(Dropout(0.15))
    model.add(layers.Dense(1,kernel_initializer='random_normal',activation='linear'))
    adam=optimizers.Adam(lr=0.0059)
    model.compile(optimizer=adam,loss='mae')
    

    return model




# best_run, best_model = optim.minimize(model=create_model,
#                                       data=data,
#                                       algo=tpe.suggest,
#                                       max_evals=3,
#                                       trials=Trials())
                                      

    



# # history 會記錄訓練的狀況，會將model.fit回傳的東西記錄下來，以下是常用的示範
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc = 'upper right')
# plt.savefig('nice_mae_18_columns.h5.png')




if __name__ == '__main__':
    X_train,Y_trian,X_val,Y_val,X_test,Y_test,predictData=data()
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=150)
    ckptCallback = ModelCheckpoint('latest_0414.h5',monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=True,mode='auto',period=1)
    model = create_model()  
    history=model.fit(X_train,Y_trian,validation_data=[X_val,Y_val],callbacks=[early_stopping_callback,ckptCallback],
    epochs=600,batch_size=512,verbose=2)
    score = model.evaluate(X_test, Y_test, batch_size=32)
    print(score)
    # plot the loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc = 'upper right')
    plt.savefig('best_0414.png')

    # predict by test dataset
    model.load_weights('latest_0414.h5')
    pred = model.predict(predictData)
    np.set_printoptions(threshold=sys.maxsize)
    print("This is prediction of 2nd round : \n",pred)

    with open('latest_0414.csv', 'w') as f:
        f.write('id,SalePrice\n')
        for i in range (len(pred)):
            f.write(str(i+1461)+','+str(float(pred[i]))+'\n')
            [np.power(2, 9), np.power(2, 10), np.power(2, 11)]
