import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformer.transformer import Time2Vector, SingleAttention, MultiAttention, TransformerEncoder

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

SEQ_LEN = 128
FILE_PATH = "data/BTCUSDT_2020.11.30-2020.12.07_ML.csv"

train_data = pd.read_csv(FILE_PATH)

withoutPrice = train_data.drop(columns=['price', 'time']).values
justPrice = train_data['price'].values

X_train, y_train = [], []
for i in range(SEQ_LEN, 100000):
    X_train.append(withoutPrice[i - SEQ_LEN:i])
    y_train.append(justPrice[i])
X_train, y_train = np.array(X_train), np.array(y_train)

model = tf.keras.models.load_model('transformer/Transformer+TimeEmbedding.hdf5',
                                   custom_objects={'Time2Vector': Time2Vector,
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder': TransformerEncoder})

X_train = X_train[:100]
y_train = y_train[:100]
#Calculate predication for training, validation and test data
train_pred = model.predict(X_train)
print("done")
###############################################################################
'''Display results'''

fig = plt.figure(figsize=(15,20))
st = fig.suptitle("Transformer + TimeEmbedding Model", fontsize=22)
st.set_y(0.92)

#Plot training data results
ax11 = fig.add_subplot(311)
ax11.plot(y_train, label='Price')
ax11.plot(train_pred, linewidth=3, label='Predicted Price')
ax11.set_title("Training Data", fontsize=18)
ax11.set_ylabel('Price')
ax11.legend(loc="best", fontsize=12)

plt.show()