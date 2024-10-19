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

salesByDate = pd.DataFrame(data, columns=dataName)
salesByDate = salesByDate[salesByDate['Order Date'].dt.year == 2019]
salesByDate = salesByDate.groupby(salesByDate['Order Date'].dt.date)['Sales'].sum()
salesByDate.plot()
plt.show()
test = salesByDate.to_frame()

print(list(test.values))

#plt.plot(x, y)
#plt.xlabel('Date')
#plt.ylabel('Sales data')
#plt.xticks(ticks=x[::60], rotation=45)
#plt.xticks([])

