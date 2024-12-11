#Explicacion
Cargamos los datos históricos de la demanda en un DataFrame de pandas.
Preparamos los datos para el RNN escalando los valores de la demanda entre 0 y 1 utilizando la clase MinMaxScaler.
Creamos una secuencia de datos para el RNN, donde cada elemento es una ventana de 12 meses de datos históricos de la demanda.
Convertimos los datos a arrays de numpy.
Creamos un modelo de RNN utilizando la clase Sequential de Keras. El modelo consta de una capa de LSTM con 50 neuronas y una capa de salida densa con una neurona.
Entrenamos el modelo utilizando la función fit y los datos de entrenamiento.
Predecimos la demanda para el próximo mes utilizando la función predict y el último elemento de la secuencia de datos.
Imprimimos el resultado.
