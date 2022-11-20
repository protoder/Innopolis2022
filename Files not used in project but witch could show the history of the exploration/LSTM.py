import numpy as np
import random
from tensorflow.keras.models import Model, Sequential, load_model  # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Dropout, Activation, Flatten, SpatialDropout2D, MaxPooling2D, \
    AveragePooling2D, Conv2D, Dense, LSTM, BatchNormalization, Reshape, Conv1D, MaxPooling1D  # Импортируем стандартные слои keras
from tensorflow.keras import backend as K  # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Adam  # Импортируем оптимизатор Adam
from tensorflow.keras import \
    utils  # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
from keras import regularizers
from keras.callbacks import Callback
import tensorflow as tf
import os
from tqdm.auto import tqdm
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras import backend as K

from Libs import * #ReadCsv, WriteCsv, Graphic, Filter
from Experiments import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def CreateLSTM(X, P0 = 200, P1 = 50, P2 = 200, P3 = 50, RS = True):
    model = Sequential()
    model.add(LSTM(P0, input_shape=(X.shape[-1], 1), return_sequences=RS))
    model.add(LSTM(P1))
    model.add(Dense(P2, activation="relu"))
    model.add(Dense(P3, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="mse", optimizer=Adam(lr=1e-5))
    model.fit(trainDataGen, epochs=20)
