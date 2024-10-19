from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.layers import GRU

salesData = pd.read_csv("Sales Data.csv", index_col=0, sep=",")
dataName = list(salesData.columns)
data = salesData.values


for row in data:
    row[4] = datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S')

data = data[data[:, 4].argsort()]


y = data[::200, 7]
x = data[::200, 4]

salesSum = [data[1:month for month in data[4], :].sum(axis=0)]
plt.plot(x, y)
plt.xlabel('Date')
plt.ylabel('Sales data')
plt.xticks(ticks=x[::60], rotation=45)
#plt.xticks([])