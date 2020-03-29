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


data = pandas.read_excel('Country_Cases.xlsx')

scaler = MinMaxScaler(feature_range=(0, 1))

country='US'
lookback=3
lookahead=2

train_dates=['J20','J21','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11']
test_dates=['M11','M12','M13','M14','M15','M16','M17','M18','M19','M20','M21','M22']

testX_total=data[data['Country']==country][test_dates[:-(lookback-1+lookahead)]].values[0]
testX=[]
for i in testX_total:
    temp=[]
    for j in range(lookback):
        temp.append(i)
    testX.append(temp)


testY=data[data['Country']==country][test_dates[(lookback-1+lookahead):]].values[0]

trainX_total=data[data['Country']==country][train_dates[:-(lookback-1+lookahead)]].values[0]
trainX=[]
for i in trainX_total:
    temp=[]
    for j in range(lookback):
        temp.append(i)
    trainX.append(temp)

print(train_dates[lookback-1+lookahead:])

trainY=data[data['Country']==country][train_dates[(lookback-1+lookahead):]].values[0]

p=np.concatenate((testX_total,testY))
p=np.concatenate((p,trainX_total))
p=np.concatenate((p,trainY))

p=p.reshape(-1,1)
scaler.fit(p)

testX=np.array(testX) 
#testX=testX.reshape(-1,1)
testX = scaler.transform(testX)

testY=testY.reshape(-1,1)
testY = scaler.transform(testY)

trainX=np.array(trainX)
#trainX=trainX.reshape(-1,1)
trainX = scaler.transform(trainX)

trainY = trainY.reshape(-1,1)
trainY = scaler.transform(trainY)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(5, input_shape=(1, lookback)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
trainY=trainY.reshape(-1,1)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
testY=testY.reshape(-1,1)

plt.figure(0)

plt.title(country+" - Learning Curve for 80% Training Data") # 80% training data write # how much data
plt.scatter(train_dates[:-(lookback-1+lookahead)],trainY)
plt.scatter(train_dates[:-(lookback-1+lookahead)],trainPredict)


plt.xlabel("Date")
plt.ylabel("Number of COVID-19 Infections")
plt.ylim(0,max(trainY)+1000) 


plt.xticks(np.arange(49), ['','','J25','','','J28','','','J31','','','F3','','','F6','','','F9','','','F12','','','F15','','','F18','','','F21','','','F24','','','F27','','','M1','','','M4','','','M7','','','M10',''])
plt.gca().legend(('actual','learnt'))
plt.savefig(country+"_train_"+"b_"+str(lookback)+"_a_"+str(lookahead)+".png")


plt.figure(1)

plt.title(country+" - Predictions for Test Data")
plt.scatter(test_dates[(lookback-1+lookahead):],testY)
plt.scatter(test_dates[(lookback-1+lookahead):],testPredict)


plt.xlabel("Date")
plt.ylabel("Number of COVID-19 Infections") # discrete rep for training and test
plt.ylim(0,max(testY)+7000) 


plt.xticks(np.arange(10), ['M13','M14','M15','M16','M17','M18','M19','M20','M21','M22'])
plt.gca().legend(('actual','predicted'))
plt.savefig(country+"_test_"+"b_"+str(lookback)+"_a_"+str(lookahead)+".png")
