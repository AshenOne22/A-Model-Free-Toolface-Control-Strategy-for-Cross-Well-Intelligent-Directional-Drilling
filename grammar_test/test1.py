import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

amodel=tf.keras.Sequential()
amodel.add(tf.keras.layers.Dense(3))
a=1
print(amodel(a))
high = np.array([1,1, 1])
state=np.random.uniform(low=-high , high=high)
print(state)
state=state.reshape(1,3)
print(state)

