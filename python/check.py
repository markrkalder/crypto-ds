import math

import tensorflow as tf

from transformer.transformer import Time2Vector, SingleAttention, MultiAttention, TransformerEncoder

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator

SEQ_LEN = 256
FILE_PATH = "data/BTCUSDT_2020.11.30-2020.12.07_ML.csv"

train_data = pd.read_csv(FILE_PATH)
train_data[['last_close_change','candle_change','high','low','RSD','REMA_9','REMA_21','REMA_50']] *= 100
train_data[['last_close_change','REMA_9','REMA_21','REMA_50']] = train_data[['last_close_change','REMA_9','REMA_21','REMA_50']].rolling(50).mean()
min_RMACD = train_data['RMACD'].min()
max_RMACD = train_data['RMACD'].max()
train_data['RMACD'] = (train_data['RMACD']- min_RMACD) / (max_RMACD - min_RMACD)
train_data['RSI'] = (train_data['RSI'] / 50) - 1
y_train = np.array(train_data['last_close_change'].values)
X_train = np.array(train_data[['RSI','RSD', 'last_close_change', 'REMA_9']].values)
X_train = X_train[:SEQ_LEN * 2]
y_train = y_train[:SEQ_LEN * 2]
data_gen = TimeseriesGenerator(X_train, y_train, SEQ_LEN, batch_size=len(X_train))


batch_0 = data_gen[0]
x,y = batch_0


model = tf.keras.models.load_model('Transformer+TimeEmbedding.hdf5',
                                   custom_objects={'Time2Vector': Time2Vector,
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder': TransformerEncoder})

#Calculate predication for training, validation and test data
train_pred = model.predict(x)
print("done")
###############################################################################
'''Display results'''

fig = plt.figure()
st = fig.suptitle("Transformer + TimeEmbedding Model", fontsize=22)

#Plot training data results
ax11 = fig.add_subplot()
ax11.plot(pd.DataFrame(y), label='Price change (9 MA)')
ax11.plot(pd.DataFrame(train_pred), linewidth=3, label='Predicted price change')
ax11.set_ylabel('Price change (9 candle avg)')
ax11.legend(loc="best", fontsize=12)

plt.show()