# Train a deep learning model for magnometry data
# input: x, y
# output: Type of the input location
# This script was adapted from: 
# https://github.com/royleekiat/Employee_attrition_predictor/blob/main/Employee_attrition_predictor_Roy_Lee.ipynb

import matplotlib.pyplot as plt
import math
import tensorflow as tf
print(tf.__version__)
import numpy as np
from numpy import unique
import pandas as pd
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

# import the file from training_1.py
dataframe = pd.read_csv('./data_training.csv')
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("Type")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def pred_dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

# features
Diam_cm = keras.Input(shape=(1,), name="Diam_cm")
Material = keras.Input(shape=(1,), name="Material")
X = keras.Input(shape=(1,), name="X")
Y = keras.Input(shape=(1,), name="Y")
Z = keras.Input(shape=(1,), name="Z")

all_inputs = [
    Diam_cm,
    Material,
    X,
    Y,
    Z
]

all_features = layers.concatenate(
[Diam_cm,
    Material,
    X,
    Y,
    Z]
)

x = layers.Dense(187, activation="tanh", name = "Dense_1")(all_features)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu", name = "Dense_2")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu", name = "Dense_3")(x)
output = layers.Dense(1, activation="sigmoid",name = "Outputlayer")(x)
model = keras.Model(all_inputs, output)
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(opt, "binary_crossentropy", metrics=["accuracy"])
model.summary()


