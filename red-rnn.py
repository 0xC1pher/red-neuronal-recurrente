import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Cargar datos históricos de la demanda
df = pd.read_csv('demanda_historica.csv')

# Preparar datos para el RNN
scaler = MinMaxScaler()
df['demanda'] = scaler.fit_transform(df['demanda'].values.reshape(-1, 1))

# Crear secuencia de datos para el RNN
X = []
y = []
for i in range(len(df) - 12):
    X.append(df['demanda'].values[i:i+12])
    y.append(df['demanda'].values[i+12])

# Convertir datos a arrays de numpy
X = np.array(X)
y = np.array(y)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Crear modelo de RNN
model = Sequential()
model.add(LSTM(50, input_shape=(12, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Predecir demanda para el próximo mes
ultima_secuencia = X_test[-1].reshape(1, 12, 1)
demanda_predicha = model.predict(ultima_secuencia)

# Invertir escala de la predicción
demanda_predicha = scaler.inverse_transform(demanda_predicha)

# Evaluar modelo
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Imprimir resultados
print('Demanda predicha para el próximo mes:', demanda_predicha[0][0])
print(f'MSE: {mse}, MAE: {mae}')
