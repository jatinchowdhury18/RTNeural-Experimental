import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Add
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../../../RTNeural')
from python.model_utils import save_model

input_shape = (32, 1)

x = Input(shape=input_shape,name = "x")
dense1 = Dense(8, activation='tanh')(x)
dense2 = Dense(8, activation='tanh')(dense1)
res = Add()([dense1, dense2])
dense_out = Dense(1, activation='linear')(res)

model = keras.Model(inputs=x, outputs=dense_out)
model.summary()

# construct signals
x_data = 10 * np.sin(np.arange(input_shape[0]) * np.pi * 0.1)
y = model.predict((x_data.reshape((1, -1, 1))))
print(y.shape)
y = y.flatten()

# save signals
np.savetxt('res_connection_x_python.csv', x_data, delimiter=',')
np.savetxt('res_connection_y_python.csv', y, delimiter=',')

save_model(model, 'res_connection.json')
