#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential,load_model

from tensorflow.keras.layers import Dense, LSTM, BatchNormalization

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

get_ipython().run_line_magic('matplotlib', 'inline')

tf.random.set_seed(2534)

def get_dataset(dataset):
    dataset = dataset
    print(dataset.shape)
    dataset=dataset.fillna(axis=0,method='ffill')
    dataset['Open.Price'] = dataset['Open.Price'].astype(int)
    dataset['Close.Price'] = dataset['Close.Price'].astype(int)
    dataset['Volume'] = dataset['Volume'].astype(int)
    dataset['SPX'] = dataset['SPX'].astype(int)
    dataset['SX5E'] = dataset['SX5E'].astype(int)
    dataset['UKX'] = dataset['UKX'].astype(int)
    dataset['VIX'] = dataset['VIX'].astype(int)
    dataframeDataset = dataset[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    ndarrayDataset = dataframeDataset.values
    return ndarrayDataset

def get_lstm_dataset(dataset, need_num, total_dataset):
    lstm_dataset = []
    for i in range(need_num, total_dataset):
        lstm_dataset.append(dataset[i-need_num:i])
    lstm_dataset = np.array(lstm_dataset)
    return lstm_dataset


for file in os.listdir('../data/Train/Train_data_2011/metal_with_every/'):
    print(file[3:-32].lower())



indices = pd.read_csv('../data/Validation/indices_all.csv',index_col=0)
yyindices=indices.loc['2018-01-02':]
sample = pd.read_csv("../data/finir_sample_submission.csv")
dic = dict.fromkeys(['nickel_60d','zinc_20d','copper_60d','copper_20d','zinc_60d','aluminium_1d','nickel_20d','nickel_1d','tin_60d','tin_20d','copper_1d','zinc_1d','lead_60d','aluminium_60d','lead_20d','aluminium_20d','lead_1d','tin_1d'])


#1d训练



for file in os.listdir('../data/Train/Train_data_2011/metal_with_every/'):   
    tt = pd.read_csv('../data/Train/Train_data_2011/metal_with_every/'+file,index_col=0)
    tt = tt[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    yy=pd.read_csv('../data/Validation/Validation_with_every/'+file[:-30]+'_validationwith_OI.csv',index_col=0)    
    ydata = pd.concat([yy,yyindices],axis=1)
    ydata = ydata[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    dataset=pd.concat([tt,ydata],axis=0)
    need_num = 20
    training_num = 1824
    ndarrayDataset = get_dataset(dataset) 
    x_train = ndarrayDataset[0:training_num] 
    y_train = []
    for i in range(0, training_num):
        y_train.append(ndarrayDataset[i,1])
    x_train, y_train = np.array(x_train), np.array(y_train)
    sc_X = MinMaxScaler(feature_range=(0, 1))
    x_train = sc_X.fit_transform(x_train)
    x_train = np.array(x_train)

    sc_Y = MinMaxScaler(feature_range=(0, 1))
    y_train = sc_Y.fit_transform(y_train.reshape(-1,1))
    y_train = np.array(y_train)
    xTrain = []
    for i in range(need_num, training_num-1):
        xTrain.append(x_train[i-need_num:i])
    xTrain = np.array(xTrain)
    x_train = xTrain
    yTrain = []
    for i in range(need_num+1, training_num):
        yTrain.append(y_train[i])
    yTrain = np.array(yTrain)
    y_train = yTrain
    model = Sequential()       
    model.add(LSTM(units=128, return_sequences=True, 
                   input_shape=[x_train.shape[1],
                                x_train.shape[2]]))
    model.add(BatchNormalization())
    model.add(LSTM(units=128))
    model.add(BatchNormalization())
    model.add(Dense(units=1))
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=100, batch_size=10)
    model.save(file[3:-32].lower()+"_1d_model.h5")


#1d预测


for file in os.listdir('../data/Train/Train_data_2011/metal_with_every/'):   
    model = load_model(file[3:-32].lower()+"_1d_model.h5")
    tt = pd.read_csv('../data/Train/Train_data_2011/metal_with_every/'+file,index_col=0)
    tt = tt[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    yy=pd.read_csv('../data/Validation/Validation_with_every/'+file[:-30]+'_validationwith_OI.csv',index_col=0)    
    ydata = pd.concat([yy,yyindices],axis=1)
    ydata = ydata[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    dataset=pd.concat([tt,ydata],axis=0)
    need_num = 20

    training_num = 1824

    ndarrayDataset = get_dataset(dataset) # (2077, 7)

    test_num = 253
    x_test = ndarrayDataset
    y_test = []

    for i in range(0, training_num+test_num):
        y_test.append(ndarrayDataset[i,1])

    x_test, y_test = np.array(x_test), np.array(y_test)

    sc_X_T = MinMaxScaler(feature_range=(0, 1))
    sc_Y_T = MinMaxScaler(feature_range=(0, 1))

    labels = []
    predictes = []
    all_time = 0
    for no in range(0,test_num+1):

        xtest = sc_X_T.fit_transform(x_test[:training_num+no])
        xtest = np.array(xtest)
        y_test_transform = sc_Y_T.fit_transform(y_test[:training_num+no].reshape(-1,1))
        y_test_transform = np.array(y_test_transform)
        xTest = []
        for i in range(training_num-100, training_num+no+1):
            xTest.append(xtest[i-need_num:i])
        xTest = np.array(xTest)
        xtest = xTest
        
        y_predictes = model.predict(x=xtest)
        
        y_predictes = sc_Y_T.inverse_transform(X=y_predictes)
        predictes.append(y_predictes[-1][0])
    end = time.time()
    all_time += end-start
    start = time.time()
    result_1d = pd.read_csv("../data/price_result/leak_label/LMEAluminium_1d_label_result.csv")
    result = predictes[-255:]
    
    print("时间为:",all_time)

    for i in range(0,test_num):

        if (result[-i] - result[-i-1]) >= 0:
            result_1d.iloc[-i-1,1] = 1.0
        else:
            result_1d.iloc[-i-1,1] = 0.0
    result_this = result_1d.set_index(["Unnamed: 0"])
    dic[file[3:-32].lower()+"_1d"]=result_this



# 20d训练



for file in os.listdir('../data/Train/Train_data_2011/metal_with_every/'):  
    indices = pd.read_csv('../data/Validation/indices_all.csv',index_col=0)
    yyindices=indices.loc['2018-01-02':]
    tt = pd.read_csv('../data/Train/Train_data_2011/metal_with_every/'+'LMEAluminium3M_trainwith_every_from_2011.csv',index_col=0)
    tt = tt[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    yy=pd.read_csv('../data/Validation/Validation_with_every/'+'LMEAluminium3M_validationwith_OI.csv',index_col=0)    
    ydata = pd.concat([yy,yyindices],axis=1)
    ydata = ydata[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    dataset=pd.concat([tt,ydata],axis=0)

    need_num = 20

    training_num = 1824

    ndarrayDataset = get_dataset(dataset) # (2077, 7)

    x_train = ndarrayDataset[0:training_num] #(1824,7)

    y_train = []

    for i in range(20, training_num):
        y_train.append(ndarrayDataset[i,1])

    x_train, y_train = np.array(x_train), np.array(y_train)


    sc_X = MinMaxScaler(feature_range=(0, 1))
    x_train = sc_X.fit_transform(x_train)
    x_train = np.array(x_train)

    sc_Y = MinMaxScaler(feature_range=(0, 1))
    y_train = sc_Y.fit_transform(y_train.reshape(-1,1))
    y_train = np.array(y_train)

    xTrain = []
    for i in range(need_num, training_num-20+1):
        xTrain.append(x_train[i-need_num:i])
    xTrain = np.array(xTrain) 

    x_train = xTrain

    yTrain = []
    for i in range(need_num, training_num-19):
        yTrain.append(y_train[i-1])
    yTrain = np.array(yTrain)

    y_train = yTrain
    
    model = Sequential()
     
    model.add(LSTM(units=128, return_sequences=True, 
                   input_shape=[x_train.shape[1],
                                x_train.shape[2]]))
    model.add(BatchNormalization())

    model.add(LSTM(units=128))
    model.add(BatchNormalization())

    model.add(Dense(units=1))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=200, batch_size=10)
    model.save(file[3:-32].lower()+"_20d_model.h5")



#20d预测



for file in os.listdir('../data/Train/Train_data_2011/metal_with_every/'):   
    model = load_model(file[3:-32].lower()+"_20d_model.h5")
    tt = pd.read_csv('../data/Train/Train_data_2011/metal_with_every/'+file,index_col=0)
    tt = tt[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    yy=pd.read_csv('../data/Validation/Validation_with_every/'+file[:-30]+'_validationwith_OI.csv',index_col=0)    
    ydata = pd.concat([yy,yyindices],axis=1)
    ydata = ydata[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    dataset=pd.concat([tt,ydata],axis=0)

    need_num = 20

    training_num = 1824

    ndarrayDataset = get_dataset(dataset) 
    test_num = 253
    x_test = ndarrayDataset
    y_test = []

    for i in range(0, training_num+test_num):
        y_test.append(ndarrayDataset[i,1])

    x_test, y_test = np.array(x_test), np.array(y_test)

    sc_X_T = MinMaxScaler(feature_range=(0, 1))
    sc_Y_T = MinMaxScaler(feature_range=(0, 1))

    labels = []
    predictes = []
    all_time=0
    start = time.time()
    for no in range(-20,test_num+1):

        xtest = sc_X_T.fit_transform(x_test[:training_num+no])
        xtest = np.array(xtest)
        y_test_transform = sc_Y_T.fit_transform(y_test[:training_num+no].reshape(-1,1))
        y_test_transform = np.array(y_test_transform)
        xTest = []
        for i in range(training_num-100, training_num+no+1):
            xTest.append(xtest[i-need_num:i])
        xTest = np.array(xTest)
        xtest = xTest
        y_predictes = model.predict(x=xtest)
        y_predictes = sc_Y_T.inverse_transform(X=y_predictes)
        predictes.append(y_predictes[-1][0])
    end = time.time()
    all_time += end-start

    result_20d = pd.read_csv("../data/price_result/leak_label/LMEAluminium_1d_label_result.csv")
    result = predictes[-273:]
    print("时间为:",all_time)
    y_test = np.roll(y_test,-20)

    theshold_price = {'aluminium':40,'copper':300,'lead':30,'nickel':600,'tin':0,'zinc':50}
    for i in range(21,test_num+21):
        if (result[-i+20] - result[-i]) > theshold_price[file[3:-32].lower()]:
            result_20d.iloc[-i+20,1] = 1.0
        else:
            result_20d.iloc[-i+20,1] = 0.0
    result_this = result_20d.set_index(["Unnamed: 0"])
    dic[file[3:-32].lower()+"_20d"]=result_this



#60d训练


for file in os.listdir('../data/Train/Train_data_2011/metal_with_every/'):  
    indices = pd.read_csv('../data/Validation/indices_all.csv',index_col=0)
    yyindices=indices.loc['2018-01-02':]
    tt = pd.read_csv('../data/Train/Train_data_2011/metal_with_every/'+'LMEAluminium3M_trainwith_every_from_2011.csv',index_col=0)
    tt = tt[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    yy=pd.read_csv('../data/Validation/Validation_with_every/'+'LMEAluminium3M_validationwith_OI.csv',index_col=0)    
    ydata = pd.concat([yy,yyindices],axis=1)
    ydata = ydata[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    dataset=pd.concat([tt,ydata],axis=0)

    need_num = 60

    training_num = 1824

    ndarrayDataset = get_dataset(dataset)

    x_train = ndarrayDataset[0:training_num]

    y_train = []

    for i in range(60, training_num):
        y_train.append(ndarrayDataset[i,1])

    x_train, y_train = np.array(x_train), np.array(y_train)

    sc_X = MinMaxScaler(feature_range=(0, 1))
    x_train = sc_X.fit_transform(x_train)
    x_train = np.array(x_train)

    sc_Y = MinMaxScaler(feature_range=(0, 1))
    y_train = sc_Y.fit_transform(y_train.reshape(-1,1))
    y_train = np.array(y_train)

    xTrain = []
    for i in range(need_num, training_num-60+1):
        xTrain.append(x_train[i-need_num:i])
    xTrain = np.array(xTrain) 

    x_train = xTrain

    yTrain = []
    for i in range(need_num, training_num-59):
        yTrain.append(y_train[i-1])
    yTrain = np.array(yTrain)

    y_train = yTrain
    model = Sequential()
      
    model.add(LSTM(units=128, return_sequences=True, 
                   input_shape=[x_train.shape[1],
                                x_train.shape[2]]))
    model.add(BatchNormalization())

    model.add(LSTM(units=128))
    model.add(BatchNormalization())

    model.add(Dense(units=1))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=200, batch_size=10)
    model.save(file[3:-32].lower()+"_60d_model.h5")




#60d预测



for file in os.listdir('../data/Train/Train_data_2011/metal_with_every/'):   
    model = load_model(file[3:-32].lower()+"_60d_model.h5")
    tt = pd.read_csv('../data/Train/Train_data_2011/metal_with_every/'+file,index_col=0)
    tt = tt[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    yy=pd.read_csv('../data/Validation/Validation_with_every/'+file[:-30]+'_validationwith_OI.csv',index_col=0)    
    ydata = pd.concat([yy,yyindices],axis=1)
    ydata = ydata[['Open.Price', 'Close.Price', 'Volume', 'SPX', 'SX5E', 'UKX', 'VIX']]
    dataset=pd.concat([tt,ydata],axis=0)

    need_num = 60

    training_num = 1824

    ndarrayDataset = get_dataset(dataset)

    test_num = 253
    x_test = ndarrayDataset
    y_test = []

    for i in range(0, training_num+test_num):
        y_test.append(ndarrayDataset[i,1])

    x_test, y_test = np.array(x_test), np.array(y_test)

    sc_X_T = MinMaxScaler(feature_range=(0, 1))
    sc_Y_T = MinMaxScaler(feature_range=(0, 1))

    labels = []
    predictes = []
    all_time = 0
    start = time.time()
    for no in range(-60,test_num+1):

        xtest = sc_X_T.fit_transform(x_test[:training_num+no])
        xtest = np.array(xtest)
        y_test_transform = sc_Y_T.fit_transform(y_test[:training_num+no].reshape(-1,1))
        y_test_transform = np.array(y_test_transform)
        xTest = []
        for i in range(training_num-100, training_num+no+1):
            xTest.append(xtest[i-need_num:i])
        xTest = np.array(xTest)
        xtest = xTest
        y_predictes = model.predict(x=xtest)
        y_predictes = sc_Y_T.inverse_transform(X=y_predictes)
        predictes.append(y_predictes[-1][0])
    end = time.time()
    all_time += end-start

    result_60d = pd.read_csv("../data/price_result/leak_label/LMEAluminium_1d_label_result.csv")
    result = predictes[-313:]
    print("时间为:",all_time)
    y_test = np.roll(y_test,-60)
    theshold_price = {'aluminium':0,'copper':1000,'lead':300,'nickel':0,'tin':5000,'zinc':300}
    for i in range(61,test_num+61):
        if (result[-i+60] - result[-i]) > theshold_price[file[3:-32].lower()]:
            result_60d.iloc[-i+60,1] = 1.0
        else:
            result_60d.iloc[-i+60,1] = 0.0
    result_this = result_60d.set_index(["Unnamed: 0"])
    dic[file[3:-32].lower()+"_60d"]=result_this



my_submission = pd.concat([dic['nickel_60d'],dic['zinc_20d'],dic['copper_60d'],dic['copper_20d'],dic['zinc_60d'],dic['aluminium_1d'],dic['nickel_20d'],dic['nickel_1d'],dic['tin_60d'],dic['tin_20d'],dic['copper_1d'],dic['zinc_1d'],dic['lead_60d'],dic['aluminium_60d'],dic['lead_20d'],dic['aluminium_20d'],dic['lead_1d'],dic['tin_1d']],axis=0)



for i in range(sample.shape[0]):
    sample.iloc[i,1] = my_submission.iloc[i,0]
sample = sample.set_index("id")
sample.to_csv(time.strftime("../data/%Y-%m-%d %H_%M_%S", time.localtime())+".csv")




