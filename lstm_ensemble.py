import pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics

data = pandas.read_excel('Confirmed_Cases_1.xlsx')

data.fillna(0)

all_counties = data['Combined_Key'].values

train_dates=['J22','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13','M14','M15','M16','M17','M18','M19','M20','M21','M22','M23','M24','M25','M26','M27','M28','M29','M30','M31','A1','A2','A3','A4','A5','A6','A7','A8']
test_dates=['A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22']

lookback = 5
lookahead = 5 
num_layers = 5

train_rmse_total = 0
test_rmse_total = 0

num_counties = 0

trainY_total = []
trainPredict_total = []
for i in range(len(train_dates[(lookback+lookahead):])):
        trainY_total.append(0)
        trainPredict_total.append(0)

testY_total = []
testPredict_total = []
testBaseline_total = []
for i in range(len(test_dates[(lookback+lookahead):])):
        testY_total.append(0)
        testPredict_total.append(0)
        testBaseline_total.append(0)

w = 0

for county in all_counties:

        scaler = MinMaxScaler(feature_range=(0, 1))

        testY=data[data['Combined_Key']==county][test_dates[(lookback+lookahead):]].values[0]

        testX_total=data[data['Combined_Key']==county][test_dates[:-(lookahead)]].values[0]

        testX=[]

        testBaseline=[] #moving average baseline
        
        for i in range(len(testY)):
            temp=[]
            for j in range(lookback):
                temp.append(testX_total[i+j])
            avg=statistics.mean(temp)
            testBaseline.append(avg)
            testX.append(temp)

        trainY=data[data['Combined_Key']==county][train_dates[(lookback+lookahead):]].values[0]

        trainX_total=data[data['Combined_Key']==county][train_dates[:-(lookahead)]].values[0]

        trainX=[]
        for i in range(len(trainY)):
            temp=[]
            for j in range(lookback):
                temp.append(trainX_total[i+j])
            trainX.append(temp)

        p=np.concatenate((testX_total,testY))
        p=np.concatenate((p,trainX_total))
        p=np.concatenate((p,trainY))

        p=p.reshape(-1,1)
        scaler.fit(p)

        testX=np.array(testX)
        if lookback==1:
            testX=testX.reshape(-1,1)

        testX = scaler.transform(testX)

        testY=testY.reshape(-1,1)
        testY = scaler.transform(testY)

        trainX=np.array(trainX)
        if lookback==1:
            trainX=trainX.reshape(-1,1)
        trainX = scaler.transform(trainX)

        trainY = trainY.reshape(-1,1)
        trainY = scaler.transform(trainY)

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        model = Sequential()
        model.add(LSTM(num_layers, input_shape=(1, lookback)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform(trainY)
        trainY=trainY.reshape(-1,1)
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform(testY)
        testY=testY.reshape(-1,1)

        train_rmse=sqrt(mean_squared_error(trainY, trainPredict))

        train_rmse_total = train_rmse_total + train_rmse

        test_rmse=sqrt(mean_squared_error(testY, testPredict))

        test_rmse_total = test_rmse_total + test_rmse

        trainY = trainY.reshape(1,-1)

        trainY_total = np.add(trainY_total, trainY[0])

        trainPredict = trainPredict.reshape(1,-1)

        trainPredict_total = np.add(trainPredict_total, trainPredict[0])

        testY = testY.reshape(1,-1)

        testY_total = np.add(testY_total, testY[0])

        testPredict = testPredict.reshape(1,-1)

        testPredict_total = np.add(testPredict_total, testPredict[0])

        testBaseline_total = np.add(testBaseline_total, testBaseline)

        num_counties = num_counties + 1

        if num_counties % 100 == 0:
                print(num_counties, " done")

f = open("test_values_1.txt", "w")
f.write(str(train_rmse_total)+"\n")
f.write(str(test_rmse_total)+"\n")
f.write(str(trainY_total)+"\n")
f.write(str(trainPredict_total)+"\n")
f.write(str(testY_total)+"\n")
f.write(str(testPredict_total)+"\n")
f.write(str(testBaseline_total)+"\n")
f.close()
