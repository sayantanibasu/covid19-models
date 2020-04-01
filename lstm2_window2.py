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


data = pandas.read_excel('Country_Cases.xlsx')

#f = open('Parameters.txt','a')

scaler = MinMaxScaler(feature_range=(0, 1))

country_all = ['US, Washington ', 'US, New York']

#'US', 'India', 'Italy', 'Spain', 'Germany', 'Iran', 'Korea, South', 'Japan', 'China, Hubei', 'UK, UK', 

lookback_all=[1,2,3,4,5]
lookahead_all=[1,2,3,4,5]
num_layers_all=[5]

#f.write("country"+"\t"+"lookback"+"\t"+"lookahead"+"\t"+"num_layers"+"\t"+"train_rmse"+"\t"+"test_rmse"+"\n")

#f.close()

w=0

train_dates=['J22','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13','M14','M15','M16']
test_dates=['M16','M17','M18','M19','M20','M21','M22','M23','M24','M25','M26','M27','M28','M29']

for country in country_all:

    for lookback in lookback_all:

        for lookahead in lookahead_all:

            for num_layers in num_layers_all:

                #f = open('Parameters.txt','a')

                testY=data[data['Country']==country][test_dates[(lookback+lookahead):]].values[0]

                testX_total=data[data['Country']==country][test_dates[:-(lookahead)]].values[0]

                testX=[]
                for i in range(len(testY)):
                    temp=[]
                    for j in range(lookback):
                        temp.append(testX_total[i+j])
                    testX.append(temp)

                trainY=data[data['Country']==country][train_dates[(lookback+lookahead):]].values[0]

                trainX_total=data[data['Country']==country][train_dates[:-(lookahead)]].values[0]

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

                model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

                trainPredict = model.predict(trainX)
                testPredict = model.predict(testX)

                trainPredict = scaler.inverse_transform(trainPredict)
                trainY = scaler.inverse_transform(trainY)
                trainY=trainY.reshape(-1,1)
                testPredict = scaler.inverse_transform(testPredict)
                testY = scaler.inverse_transform(testY)
                testY=testY.reshape(-1,1)

                plt.figure(w)

                plt.title(country+" - Learning Curve for 80% Training Data") # 80% training data write # how much data
                plt.scatter(train_dates[(lookback+lookahead):],trainY)
                plt.scatter(train_dates[(lookback+lookahead):],trainPredict)

                train_rmse=sqrt(mean_squared_error(trainY, trainPredict))

                plt.xlabel("Date")
                plt.ylabel("Number of COVID-19 Infections")
                plt.ylim(0,1.2*max(trainY)) 

                ticks_x=[]
                j=0
                for i in train_dates[(lookback+lookahead):]:
                    if j%4==0:
                        ticks_x.append(i)
                    else:
                        ticks_x.append(' ')
                    j=j+1


                plt.xticks(np.arange(len(train_dates[(lookback+lookahead):])), ticks_x)
                plt.gca().legend(('actual','learnt'))
                plt.figtext(1, 1, 'lookback : '+str(lookback)+', lookahead : '+str(lookahead)+', num_layers : '+str(num_layers)+', Train RMSE : '+str(train_rmse), horizontalalignment='right')
                plt.savefig(country+"_train_"+"b_"+str(lookback)+"_a_"+str(lookahead)+"_n_"+str(num_layers)+".png", bbox_inches='tight')

                plt.figure(w+1)

                plt.title(country+" - Predictions for Test Data")
                plt.scatter(test_dates[(lookback+lookahead):],testY)
                plt.scatter(test_dates[(lookback+lookahead):],testPredict)

                test_rmse=sqrt(mean_squared_error(testY, testPredict))


                plt.xlabel("Date")
                plt.ylabel("Number of COVID-19 Infections") # discrete rep for training and test
                plt.ylim(0,1.2*max(testY)) 

                plt.xticks(np.arange(len(test_dates[(lookback+lookahead):])), test_dates[(lookback+lookahead):])
                plt.gca().legend(('actual','predicted'))
                plt.figtext(1, 1, 'lookback : '+str(lookback)+', lookahead : '+str(lookahead)+', num_layers : '+str(num_layers)+', Test RMSE : '+str(test_rmse), horizontalalignment='right')
                plt.savefig(country+"_test_"+"b_"+str(lookback)+"_a_"+str(lookahead)+"_n_"+str(num_layers)+".png", bbox_inches='tight')

                #f.write(str(country)+"\t"+str(lookback)+"\t"+str(lookahead)+"\t"+str(num_layers)+"\t"+str(train_rmse)+"\t"+str(test_rmse)+"\n")

                #f.close()

                w=w+2
