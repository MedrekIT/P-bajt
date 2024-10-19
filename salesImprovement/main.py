from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.layers import GRU

salesData = pd.read_csv("Sales Data.csv", index_col=0, sep=",")
dataName = list(salesData.columns)
data = salesData.values

for row in data:
    row[4] = datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S')
    

salesByDate = pd.DataFrame(data, columns=dataName)
salesByDate['Order Date'] = pd.to_datetime(salesByDate['Order Date'])
salesByDate = salesByDate[salesByDate['Order Date'].dt.year == 2019]

salesByDate = salesByDate.groupby(salesByDate['Order Date'].dt.date)['Sales'].sum()

salesByDate.plot()
plt.show()

data = pd.DataFrame(data, columns=dataName)
data['Year'] = data['Order Date'].dt.year
data['Month'] = data['Order Date'].dt.month
data['Day'] = data['Order Date'].dt.day
data['DayOfWeek'] = data['Order Date'].dt.dayofweek
data['Hour'] = data['Order Date'].dt.hour

label_encoder = LabelEncoder()
data['Product Code'] = label_encoder.fit_transform(data['Product'])

data = pd.get_dummies(data, columns=['City'], drop_first=True)

data_grouped = data[data['Order Date'].dt.year == 2019]
data_grouped = data_grouped.groupby(data_grouped['Order Date'].dt.date)['Sales'].sum().reset_index()


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_grouped[['Sales']])

print(data_grouped)

salesByDate = salesByDate.to_frame().reset_index()
salesSumData = salesByDate.values
salesSumCols = list(salesByDate.columns)


#plt.plot(x, y)
#plt.xlabel('Date')
#plt.ylabel('Sales data')
#plt.xticks(ticks=x[::60], rotation=45)
#plt.xticks([])

