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
from matplotlib.cm import get_cmap
import plotly.figure_factory as ff
import plotly.express as px

data = pandas.read_excel('County_Data_Illinois.xlsx')

#f = open('Parameters.txt','a') 


dates=['J22','J23','J24','J25','J26','J27','J28','J29','J30','J31','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13','M14','M15','M16','M17','M18','M19','M20','M21','M22','M23','M24','M25','M26','M27','M28','M29','M30','M31','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16']

all_counties=data['County'].values

total=[]

for i in range(len(dates)):
    total.append(0)

for county in all_counties:
    infections=data[data['County']==county][dates].values[0]
    total=total+infections

print(total)

plt.figure(0)

all_counties_plotted=[]
infections_final=[]

for county in all_counties:
    infections=data[data['County']==county][dates].values[0]
    print(np.corrcoef(infections,total)[0][1])
    if np.corrcoef(infections,total)[0][1]>0.99:
        plt.scatter(range(len(dates)),infections)
        plt.text(len(dates)-1,infections[len(dates)-1],county,fontsize=8)
        infections_final.append(infections[len(dates)-1])
        all_counties_plotted.append(county)
        print(np.corrcoef(infections,total)[0][1])

plt.scatter(range(len(dates)),total)
plt.text(len(dates)-1,total[len(dates)-1],'New York (state)',fontsize=8)

all_counties_plotted.append('Illinois')

plt.gca().legend((all_counties_plotted),loc='upper left', ncol=2)
plt.xlim(0,1.2*len(dates)) 
plt.ylim(0,1.2*max(total)) 
plt.title("County-wise Correlation for Illinois for correlation>0.99")
plt.xlabel("Number of Days")
plt.ylabel("Confirmed Cases")

plt.show()

df_sample = pandas.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')
df_sample_r = df_sample[df_sample['STNAME'] == 'Illinois']

fips = ['17011',
        '17031',
        '17043',
        '17063',
        '17075',
        '17085',
        '17089',
        '17091',
        '17093',
        '17097',
        '17099',
        '17105',
        '17111',
        '17113',
        '17119',
        '17163',
        '17167',
        '17195',
        '17197',
        '17201',
        ]

values = infections_final

endpts = list(np.mgrid[min(values):max(values):9j])
#colorscale = ['lightcoral','coral','crimson','darkred','darksalmon','darkviolet','chocolate']
#colorscale=cmap('Reds')
#print(px.colors.sequential.Plasma)
fig = ff.create_choropleth(
    fips=fips, values=values, scope=['Illinois'], show_state_data=True,
    colorscale=px.colors.qualitative.Prism, binning_endpoints=endpts, round_legend_values=True,
    plot_bgcolor='rgb(229,229,229)',
    paper_bgcolor='rgb(229,229,229)',
    legend_title='Infections for Illinois Counties with Correlation > 0.99',
    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
    exponent_format=True,
)
fig.layout.template = None
fig.show()
    
