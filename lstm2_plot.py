import pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from math import sqrt, log, exp
import statistics

data = pandas.read_excel('Country_Cases.xlsx')

country_all = ['Cook, Illinois']

#'US', 'India', 'Italy', 'Spain', 'Germany', 'Iran', 'Korea, South', 'Japan', 'China, Hubei', 'UK, UK', 'US, Washington ', 'US, New York'

lookback_lookahead=[(1,5),(1,6),(1,7),(2,5),(2,6),(2,7),(3,5),(3,6),(3,7),(4,5),(4,6),(4,7),(5,5),(5,6),(6,5)]
num_layers=10

lookback_opti=1
lookahead_opti=7

w=0

valid_rmse_min=99999999999999999999

train_dates1=['J22','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25']


valid_dates1=['F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7']


train_dates2=['J22','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7']


valid_dates2=['M7','M8','M9','M10','M11','M12','M13','M14','M15','M16','M17','M18','M19','M20','M21','M22']


train_dates3=['J22','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13','M14','M15','M16','M17','M18','M19','M20','M21','M22']


valid_dates3=['M22','M23','M24','M25','M26','M27','M28','M29','M30','M31','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11']


train_dates=['J22','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13','M14','M15','M16','M17','M18','M19','M20','M21','M22','M23','M24','M25','M26','M27','M28','M29','M30','M31','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11']


test_dates=['A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24','A25','A26','A27','A28','A29','A30','MA1','MA2','MA3','MA4']


for country in country_all:

    for lookback,lookahead in lookback_lookahead:

        model1 = Sequential()
        model1.add(LSTM(num_layers, input_shape=(1, lookback)))
        model1.add(Dense(1))
        model1.compile(loss='mean_squared_error', optimizer='adam')

        print(model1.optimizer.get_config())

        model2 = Sequential()
        model2.add(LSTM(num_layers, input_shape=(1, lookback)))
        model2.add(Dense(1))
        model2.compile(loss='mean_squared_error', optimizer='adam')

        print(model2.optimizer.get_config())

        model3 = Sequential()
        model3.add(LSTM(num_layers, input_shape=(1, lookback)))
        model3.add(Dense(1))
        model3.compile(loss='mean_squared_error', optimizer='adam')

        print(model3.optimizer.get_config())

        validY1=data[data['Country']==country][valid_dates1[(lookback+lookahead):]].values[0]

        validX1_total=data[data['Country']==country][valid_dates1[:-(lookahead)]].values[0]

        validX1=[]

        for i in range(len(validY1)):
            temp=[]
            for j in range(lookback):
                temp.append(validX1_total[i+j])
            validX1.append(temp)

        trainY1=data[data['Country']==country][train_dates1[(lookback+lookahead):]].values[0]

        trainX1_total=data[data['Country']==country][train_dates1[:-(lookahead)]].values[0]

        trainX1=[]
        for i in range(len(trainY1)):
            temp=[]
            for j in range(lookback):
                temp.append(trainX1_total[i+j])
            trainX1.append(temp)

        validX1=np.array(validX1)
        if lookback==1:
            validX1=validX1.reshape(-1,1)

        for i in range(len(validX1)):
            validX1[i][0] = log(1+validX1[i][0])

        validY1=validY1.reshape(-1,1)

        for i in range(len(validY1)):
            validY1[i][0] = log(1+validY1[i][0])
            
        trainX1=np.array(trainX1)
        if lookback==1:
            trainX1=trainX1.reshape(-1,1)

        for i in range(len(trainX1)):
            trainX1[i][0] = log(1+trainX1[i][0])
            
        trainY1 = trainY1.reshape(-1,1)

        for i in range(len(trainY1)):
            trainY1[i][0] = log(1+trainY1[i][0])
            
        trainX1 = np.reshape(trainX1, (trainX1.shape[0], 1, trainX1.shape[1]))
        validX1 = np.reshape(validX1, (validX1.shape[0], 1, validX1.shape[1]))

        model1.fit(trainX1, trainY1, epochs=10000, batch_size=10, verbose=0)
        
        validPredict1 = model1.predict(validX1)

        valid_rmse1=sqrt(mean_squared_error(validY1, validPredict1))

        print("Validation Set 1 Completed")

        ############################################################################

        validY2=data[data['Country']==country][valid_dates2[(lookback+lookahead):]].values[0]

        validX2_total=data[data['Country']==country][valid_dates2[:-(lookahead)]].values[0]

        validX2=[]

        for i in range(len(validY2)):
            temp=[]
            for j in range(lookback):
                temp.append(validX2_total[i+j])
            validX2.append(temp)

        trainY2=data[data['Country']==country][train_dates2[(lookback+lookahead):]].values[0]

        trainX2_total=data[data['Country']==country][train_dates2[:-(lookahead)]].values[0]

        trainX2=[]
        for i in range(len(trainY2)):
            temp=[]
            for j in range(lookback):
                temp.append(trainX2_total[i+j])
            trainX2.append(temp)

        validX2=np.array(validX2)
        if lookback==1:
            validX2=validX2.reshape(-1,1)

        for i in range(len(validX2)):
            validX2[i][0] = log(1+validX2[i][0])

        validY2=validY2.reshape(-1,1)

        for i in range(len(validY2)):
            validY2[i][0] = log(1+validY2[i][0])
            
        trainX2=np.array(trainX2)
        if lookback==1:
            trainX2=trainX2.reshape(-1,1)

        for i in range(len(trainX2)):
            trainX2[i][0] = log(1+trainX2[i][0])
            
        trainY2 = trainY2.reshape(-1,1)

        for i in range(len(trainY2)):
            trainY2[i][0] = log(1+trainY2[i][0])
            
        trainX2 = np.reshape(trainX2, (trainX2.shape[0], 1, trainX2.shape[1]))
        validX2 = np.reshape(validX2, (validX2.shape[0], 1, validX2.shape[1]))

        model2.fit(trainX2, trainY2, epochs=10000, batch_size=10, verbose=0)
        
        validPredict2 = model2.predict(validX2)

        valid_rmse2=sqrt(mean_squared_error(validY2, validPredict2))

        print("Validation Set 2 Completed")

        ############################################################################

        validY3=data[data['Country']==country][valid_dates3[(lookback+lookahead):]].values[0]

        validX3_total=data[data['Country']==country][valid_dates3[:-(lookahead)]].values[0]

        validX3=[]

        for i in range(len(validY3)):
            temp=[]
            for j in range(lookback):
                temp.append(validX3_total[i+j])
            validX3.append(temp)

        trainY3=data[data['Country']==country][train_dates3[(lookback+lookahead):]].values[0]

        trainX3_total=data[data['Country']==country][train_dates3[:-(lookahead)]].values[0]

        trainX3=[]
        for i in range(len(trainY3)):
            temp=[]
            for j in range(lookback):
                temp.append(trainX3_total[i+j])
            trainX3.append(temp)

        validX3=np.array(validX3)
        if lookback==1:
            validX3=validX3.reshape(-1,1)

        for i in range(len(validX3)):
            validX3[i][0] = log(1+validX3[i][0])

        validY3=validY3.reshape(-1,1)

        for i in range(len(validY3)):
            validY3[i][0] = log(1+validY3[i][0])
            
        trainX3=np.array(trainX3)
        if lookback==1:
            trainX3=trainX3.reshape(-1,1)

        for i in range(len(trainX3)):
            trainX3[i][0] = log(1+trainX3[i][0])
            
        trainY3 = trainY3.reshape(-1,1)

        for i in range(len(trainY3)):
            trainY3[i][0] = log(1+trainY3[i][0])
            
        trainX3 = np.reshape(trainX3, (trainX3.shape[0], 1, trainX3.shape[1]))
        validX3 = np.reshape(validX3, (validX3.shape[0], 1, validX3.shape[1]))

        model3.fit(trainX3, trainY3, epochs=10000, batch_size=10, verbose=0)
        
        validPredict3 = model3.predict(validX3)

        valid_rmse3=sqrt(mean_squared_error(validY3, validPredict3))

        print("Validation Set 3 Completed")

        valid_rmse_total=valid_rmse1+valid_rmse2+valid_rmse3

        if valid_rmse_total<=valid_rmse_min:
            valid_rmse_min=valid_rmse_total
            lookback_opti=lookback
            lookahead_opti=lookahead

    print("optimal lookback: ",lookback_opti)
    print("optimal lookahead: ",lookahead_opti)


    ####################################################################################################

    lookback=lookback_opti
    lookahead=lookahead_opti

    model = Sequential()
    model.add(LSTM(num_layers, input_shape=(1, lookback)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.optimizer.get_config())


    testY=data[data['Country']==country][test_dates[(lookback+lookahead):]].values[0]

    testX_total=data[data['Country']==country][test_dates[:-(lookahead)]].values[0]

    testX=[]

    testBaseline=[] #moving average baseline
    
    for i in range(len(testY)):
        temp=[]
        for j in range(lookback):
            temp.append(testX_total[i+j])
        avg=statistics.mean(temp)
        testBaseline.append(avg)
        testX.append(temp)

    trainY=data[data['Country']==country][train_dates[(lookback+lookahead):]].values[0]

    trainX_total=data[data['Country']==country][train_dates[:-(lookahead)]].values[0]

    trainX=[]
    for i in range(len(trainY)):
        temp=[]
        for j in range(lookback):
            temp.append(trainX_total[i+j])
        trainX.append(temp)


    testX=np.array(testX)
    if lookback==1:
        testX=testX.reshape(-1,1)

    for i in range(len(testX)):
        testX[i][0] = log(1+testX[i][0])


    testY=testY.reshape(-1,1)

    for i in range(len(testY)):
        testY[i][0] = log(1+testY[i][0])
        

    trainX=np.array(trainX)
    if lookback==1:
        trainX=trainX.reshape(-1,1)

    for i in range(len(trainX)):
        trainX[i][0] = log(1+trainX[i][0])
        

    trainY = trainY.reshape(-1,1)

    for i in range(len(trainY)):
        trainY[i][0] = log(1+trainY[i][0])
        

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model.fit(trainX, trainY, epochs=10000, batch_size=10, verbose=0)
    
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    for i in range(len(trainPredict)):
        trainPredict[i][0] = exp(trainPredict[i][0])-1
        

    for i in range(len(trainY)):
        trainY[i][0] = exp(trainY[i][0])-1

    trainY=trainY.reshape(-1,1)

    for i in range(len(testPredict)):
        testPredict[i][0] = exp(testPredict[i][0])-1
        

    for i in range(len(testY)):
        testY[i][0] = exp(testY[i][0])-1
        
    testY=testY.reshape(-1,1)

    
    plt.figure(w)

    plt.title(country+" - Learning Curve for Training Data") 
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
    plt.scatter(test_dates[(lookback+lookahead):],testBaseline)

    testBaseline=np.array(testBaseline)

    print(testBaseline)

    testPredict=testPredict.reshape(1,-1)[0]

    print(testPredict)
    

    var_baseline=np.var(testBaseline)

    var_predict=np.var(testPredict)
    
    test_rmse=sqrt(mean_squared_error(testY, testPredict))


    plt.xlabel("Date")
    plt.ylabel("Number of COVID-19 Infections") 
    plt.ylim(0,1.2*max(max(testY),max(testPredict))) 

    plt.xticks(np.arange(len(test_dates[(lookback+lookahead):])), test_dates[(lookback+lookahead):])
    plt.gca().legend(('actual','predicted','baseline'))
    plt.axvline(x='MA1', color='r')
    plt.figtext(1, 1, 'lookback : '+str(lookback)+', lookahead : '+str(lookahead)+', num_layers : '+str(num_layers)+', Test RMSE : '+str(test_rmse), horizontalalignment='right')
    plt.savefig(country+"_test_"+"b_"+str(lookback)+"_a_"+str(lookahead)+"_n_"+str(num_layers)+".png", bbox_inches='tight')
    

    print(lookback, lookahead, train_rmse, test_rmse, var_baseline, var_predict)

    w=w+2
