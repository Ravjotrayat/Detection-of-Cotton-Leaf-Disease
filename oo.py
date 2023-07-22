import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(1)

train_datagen = ImageDataGenerator(rescale = 1.0/255,shear_range = 0.2,zoom_range = 0.5,horizontal_flip = True,
                                   rotation_range=10,width_shift_range=0.2,brightness_range=[0.2,1.2]
                                        )
valid_datagen=ImageDataGenerator(rescale=1.0/255)

test_datagen=ImageDataGenerator(rescale=1.0/255)


train_data = train_datagen.flow_from_directory("C:/users/RAVJOT SINGH RAYAT/Desktop/5th sem/Cotton Disease/train",target_size = (250,250),
                                                    batch_size = 64,class_mode = 'categorical')
val_data = valid_datagen.flow_from_directory("C:/users/RAVJOT SINGH RAYAT/Desktop/5th sem/Cotton Disease/val",
                                                  target_size = (250,250),
                                                    batch_size = 64,
                                                    class_mode = 'categorical')
test_data = test_datagen.flow_from_directory("C:/users/RAVJOT SINGH RAYAT/Desktop/5th sem/Cotton Disease/test",
                                                  target_size = (250,250),
                                                    batch_size = 64,
                                                    class_mode = 'categorical')

model = keras.Sequential([ layers.InputLayer(input_shape=(250,250,3)),
                          layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                         padding='same'),
                        layers.MaxPool2D(),
                        layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
                          
                        layers.MaxPool2D(),
                        layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
                        layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
                        layers.MaxPool2D(),
                        layers.Flatten(),
                        layers.Dense(8, activation='relu'),layers.Dense(4, activation='softmax'),
                     ])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

early_stopping=keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=2
)

#Training
history=model.fit(
    train_data,
    validation_data=val_data,
    callbacks=[early_stopping],
    epochs=10,
    verbose=2
)