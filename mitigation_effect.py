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

'''
#US

a = [656039.06, 679500.25 ,703176.5, 726702.6 , 745664.75 ,763595.25 ,782683.6 , 802149.7 , 821566.94 , 845569.7 , 866973.44 , 884757.9 , 899021.7 , 914380.06 , 931401.44]

b = [600485.4 , 614680.4 , 626667. ,  639063.5 , 651768.6 , 665137.2 , 678400.8 , 691365.56 , 701665.9 , 711290.3 , 721415.4 , 731618.94 , 741676.1 , 753950.94 , 764754.75]#

#New York, New York

a=[129211.016 , 133833.14,
 137812.98 , 141552.17 , 144288.73 , 146446.12 , 148993.11 , 151638.2,
 154950.53 , 159132.84 , 163296.86 , 166097.64 , 168073.52 , 169704.61,
 171903.14 ]

b=[110758.15 , 113728.35 , 115844.56 , 118015.56 , 122507.85 , 125218.62 , 127534.04,
 129694.84 , 131267.6 , 132502.95 , 133956.23 , 135460.08 , 137335.8 , 139693.05,
 142028.42]
'''
#Cook, Illinois


a = [17003.525 , 17570.533,
 18497.998 , 19196.63 ,  19796.436 , 20354.867 , 21070.527 , 21956.82,
 22760.95 ,  23881.406 , 24755.166 , 25654.58 ,  26456.457 , 27309.848,
 28110.812 ]

b = [17638.785 , 18516.885 , 19161.857 , 19760.463 , 20433.875 , 20954.875 , 21799.059,
 22428.604 , 22965.041 , 23461.217 , 24092.488 , 24867.541 , 25564.625 , 26526.492, 
 27269.377]


y = []
x = []

for i in range(len(a)):
    y.append(a[i]-b[i])
    x.append(i)

del_y = []
del_x = []

for i in range(len(y)-1):
    del_y.append(y[i+1]-y[i])
    del_x.append(i)

print(del_y)

#plt.show()
    

m, c = np.polyfit(del_x, del_y, 1)

y_fit = []

for i in del_x:
    y_fit.append(m*i+c)

plt.xlabel("Days")
plt.ylabel("Rate of COVID-19 Infections")
plt.title("Cook, Illinois")
plt.scatter(del_x, del_y)

#plt.scatter(x, y)
plt.plot(del_x, y_fit,'r')

plt.show()
