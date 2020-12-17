import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.995))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


from transformer import Time2Vector, TransformerEncoder
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Concatenate, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator
from wandb.keras import WandbCallback

import pandas as pd
import numpy as np
import wandb

FILE_PATH = "C:/Users/marku/Desktop/Projects/crypto-ds/python/data/BTCUSDT_2019.01.01-2020.01.01_ML.csv"

BATCH_SIZE = 32
EPOCHS = 35
SEQ_LEN = 256
LR = 0.0001

D_K = 256
D_V = 256
N_HEADS = 16
FF_DIM = 256


def create_model():
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(SEQ_LEN)
    attn_layer1 = TransformerEncoder(D_K, D_V, N_HEADS, FF_DIM)
    attn_layer2 = TransformerEncoder(D_K, D_V, N_HEADS, FF_DIM)
    attn_layer3 = TransformerEncoder(D_K, D_V, N_HEADS, FF_DIM)

    '''Construct model'''
    in_seq = Input(shape=(SEQ_LEN, 4))
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='linear')(x)

    tr_model = Model(inputs=in_seq, outputs=out)
    tr_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=LR), metrics=['mae', 'mape'])
    return tr_model


train_data = pd.read_csv(FILE_PATH)
train_data[['last_close_change','candle_change','high','low','RSD','REMA_9','REMA_21','REMA_50']] *= 100
train_data[['last_close_change','REMA_9','REMA_21','REMA_50']] = train_data[['last_close_change','REMA_9','REMA_21','REMA_50']].rolling(50).mean()
min_RMACD = train_data['RMACD'].min()
max_RMACD = train_data['RMACD'].max()
train_data['RMACD'] = (train_data['RMACD']- min_RMACD) / (max_RMACD - min_RMACD)
train_data['RSI'] = (train_data['RSI'] / 50) - 1
y_train = np.array(train_data['RSI'].values)
X_train = np.array(train_data[['RSI','RSD', 'last_close_change', 'REMA_9']].values)
X_train = X_train[:SEQ_LEN * 8]
y_train = y_train[:SEQ_LEN * 8]

data_gen = TimeseriesGenerator(X_train, y_train, SEQ_LEN, batch_size=BATCH_SIZE)

wandb.init(project="crypto-ds")
config = wandb.config
config.BATCH_SIZE = BATCH_SIZE
config.SEQ_LEN = SEQ_LEN
config.D_K = D_K
config.D_V = D_V
config.N_HEADS = N_HEADS
config.FF_DIM = FF_DIM

model = create_model()
model.summary()

callback = tf.keras.callbacks.ModelCheckpoint('../Transformer+TimeEmbedding.hdf5',
                                              monitor='loss',
                                              save_best_only=True, verbose=1)

history = model.fit(data_gen, steps_per_epoch=len(data_gen), epochs=EPOCHS, callbacks=[callback, WandbCallback()])
