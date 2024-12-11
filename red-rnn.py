import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Cargar datos históricos de la demanda
df = pd.read_csv('demanda_historica.csv')

# Preparar datos para el RNN
scaler = MinMaxScaler()
df['demanda'] = scaler.fit_transform(df['demanda'])

# Crear secuencia de datos para el RNN
X = []
y = []
for i in range(len(df) - 12):
    X.append(df['demanda'].values[i:i+12])
    y.append(df['demanda'].values[i+12])

# Convertir datos a arrays de numpy
X = np.array(X)
y = np.array(y)

# Crear modelo de RNN
model = Sequential()
model.add(LSTM(50, input_shape=(12, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar modelo
model.fit(X, y, epochs=100, batch_size=32)

# Predecir demanda para el próximo mes
demanda_predicha = model.predict(X[-1].reshape(1, 12, 1))

# Imprimir resultado
print('Demanda predicha para el próximo mes:', demanda_predicha)
