import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def create_dataset(dataset, timestep=60):
    dataX, dataY = [], []
    for i in range(len(dataset) - timestep - 1):
        a = dataset[i:(i + timestep), 0]
        dataX.append(a)
        dataY.append(dataset[i + timestep, 0])
    return np.array(dataX), np.array(dataY)

def train_and_save_model():
    dtype = {'StateHoliday': str, 'SchoolHoliday': str}
    df = pd.read_csv('src/data/train.csv', dtype=dtype, parse_dates=['Date'], low_memory=False)
    store_df = pd.read_csv('src/data/store.csv')
    df = pd.merge(df, store_df, on='Store', how='left')

    store_id = 1
    data = df[df['Store'] == store_id].sort_values('Date')
    data.tail(90).to_csv('src/data/recentdata.csv', index=False)

    sales_data = data['Sales'].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(sales_data.reshape(-1, 1))

    X, y = create_dataset(scaled_data, 60)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=64, epochs=3, verbose=1)

    model.save('src/salesmodel.keras')
    with open('src/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Model trained and saved!")
