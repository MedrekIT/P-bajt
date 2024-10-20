import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout

# Preparing data to be sequential (compatible with GRU model)
def createSequences(data, seqLen):
    x = []
    y = []
    for i in range(len(data) - seqLen):
        x.append(data[i:i + seqLen]) #Sequences
        y.append(data[i + seqLen, 0])  #Results
    return np.array(x), np.array(y)

#Import data
salesData = pd.read_csv("Sales Data.csv", index_col=0, sep=",")
salesData['Order Date'] = pd.to_datetime(salesData['Order Date'])

#One-hot encoding for product names and merge encoded columns with the original data
ohEncode = pd.get_dummies(salesData['Product'], prefix='Product') #Encoding
salesDataEnc = pd.concat([salesData, ohEncode], axis=1) #Merging with dataset

#Group by date
#Sum sales, quantities, and product columns
#Use 'first' to keep the date
dataGroup = salesDataEnc[salesDataEnc['Order Date'].dt.year == 2019]
dataGroup = dataGroup.groupby(dataGroup['Order Date'].dt.date).agg({
    'Order Date': 'first',
    'Sales': 'sum',
    'Quantity Ordered': 'sum',
    **{col: 'sum' for col in ohEncode.columns}  #Sum of product columns
}).reset_index(drop=True)

#Prepare features and sales for model
features = dataGroup[['Quantity Ordered'] + list(ohEncode.columns)].values
sales = dataGroup['Sales'].values.reshape(-1, 1)

#Normalize
scaler = MinMaxScaler()
scaledFeat = scaler.fit_transform(features)
scaledSal = scaler.fit_transform(sales)

#Prepare sequences compatible with GRU
sequenceLen = 3
x, y = createSequences(np.hstack((scaledSal, scaledFeat)), sequenceLen)

#Split data into training and test
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.7, shuffle=False)

#Build GRU model
model = Sequential()

#First layer
model.add(GRU(units=100, return_sequences=True, input_shape=(xTrain.shape[1], xTrain.shape[2])))
model.add(Dropout(0.2)) #Dropout to prevent overfitting

#Second layer
model.add(GRU(units=100))
model.add(Dropout(0.2)) #Dropout to prevent overfitting

#Dense output layer
model.add(Dense(1))

#Compile and train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain, epochs=30, batch_size=64)

#Make predictions and inverse scale results
yPred = model.predict(xTest)
yPredResc = scaler.inverse_transform(yPred)
yTestResc = scaler.inverse_transform(yTest.reshape(-1, 1))

#Visualize results
plt.figure(figsize=(12, 6))
plt.plot(yTestResc, label='True Values', color='#3f407e')
plt.plot(yPredResc, label='Predictions', color='white')
plt.gca().set_facecolor("#040a44")
plt.xlabel('Data Points')
plt.ylabel('Total Sales')
plt.legend()
plt.title('True Sales vs. Predicted Sales')
plt.show()

#Evaluate model
mae = mean_absolute_error(yTest, yPred)
print(f'MAE (Mean absolute error) = {(mae * 100):.2f}%')
