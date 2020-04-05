import pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
import statistics

data = pandas.read_excel('Country_Cases.xlsx')

#f = open('Parameters.txt','a') 

country = 'China, Hubei'

#'US', 'India', 'Italy', 'Spain', 'Germany', 'Iran', 'Korea, South', 'Japan', 'China, Hubei', 'UK, UK', 'US, Washington ', 'US, New York'

lookback_all=[4]   #1,2,3,4,5
lookahead_all=[2]  #1,2,3,4,5
num_layers_all=[5]

#f.write("country"+"\t"+"lookback"+"\t"+"lookahead"+"\t"+"num_layers"+"\t"+"train_rmse"+"\t"+"test_rmse"+"\n")

#f.close()

w=0

dates=['J22','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13','M14','M15','M16','M17','M18','M19','M20','M21','M22','M23','M24','M25','M26','M27','M28','M29']

infections=data[data['Country']==country][dates].values[0]

graph=[]

for i in range(len(infections)-1):
    graph.append(infections[i+1]-infections[i])

peak=graph.index(max(graph))

x=[]
y=[]

for i in range(1,peak+1):
    if graph[i]!=0:
        x.append(graph[i])
        y.append(peak-i)

x=np.array(x)
y=np.array(y)

#print(x)
#print(y)

#model = LinearRegression()
#model.fit(x, y)

z = np.polyfit(x, y, 3)
p = np.poly1d(z)

print(x)
print(y)
print(p(x))

#print(z)

r_2 = r2_score(y, p(x))

print(r_2)
print(p)

print("US New York Peak Prediction of Infections (Number of Days) from April 3:",int(round(p(5350))))

#plt.plot(range(0,len(y)),p(x))
plt.figure(w)
plt.scatter(y[::-1],x)
plt.scatter(np.around(p(x))[::-1] ,x)
plt.gca().legend(('actual','predicted'))
a=np.arange(max(y)+1)
plt.xticks(a,max(y)-a)
plt.title("Peak Infection Prediction Model on China, Hubei")
plt.xlabel("Number of days till Peak Infections (in a day)")
plt.ylabel("Number of Infections per day")
plt.show()

#################################

w=w+1

country= 'China, Hubei Deaths'

dates=['J22','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13','M14','M15','M16','M17','M18','M19','M20','M21','M22','M23','M24','M25','M26','M27','M28','M29']

infections=data[data['Country']==country][dates].values[0]

graph=[]

for i in range(len(infections)-1):
    graph.append(infections[i+1]-infections[i])

peak=graph.index(max(graph))

x=[]
y=[]

for i in range(1,peak+1):
    if graph[i]!=0:
        x.append(graph[i])
        y.append(peak-i)

x=np.array(x)
y=np.array(y)

#print(x)
#print(y)

#model = LinearRegression()
#model.fit(x, y)

z = np.polyfit(x, y, 4)
p = np.poly1d(z)

print(x)
print(y)
print(p(x))

#print(z)

r_2 = r2_score(y, p(x))

print(r_2)
print(p)

print("US New York Peak Prediction of Deaths (Number of Days) from April 3:",int(round(p(187))))

#plt.plot(range(0,len(y)),p(x))
plt.figure(w)
plt.scatter(y[::-1],x)
plt.scatter(np.around(p(x))[::-1] ,x)
plt.gca().legend(('actual','predicted'))
a=np.arange(max(y)+1)
plt.xticks(a,max(y)-a)
plt.title("Peak Death Prediction Model on China, Hubei")
plt.xlabel("Number of days till Peak Deaths (in a day)")
plt.ylabel("Number of Deaths per day")
plt.show()


