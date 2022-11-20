import numpy as np
import random
from tensorflow.keras.models import Model, Sequential, load_model  # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Dropout, Activation, Flatten, SpatialDropout2D, MaxPooling2D, \
    AveragePooling2D, Conv2D, Dense, LSTM, GRU, BatchNormalization, Reshape, Conv1D, MaxPooling1D  # Импортируем стандартные слои keras
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

class TNN(TML_NN):
    def __init__(self, Epochs=20, Layers = {}, verbose = 1, ModelsPath = './',
                 Batch_size=64, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):

        self.ModelsPath = ModelsPath
        self.FitParams = {'epochs': Epochs, 'batch_size': Batch_size, 'verbose': verbose}
        TML_NN.__init__(self, ML_NN=None, FitParams= self.FitParams)
        self.verbose = verbose

        self.Layers = Layers
        self.Epochs = Epochs
        self.Batch_size = Batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self.Descr = loss + ' ' + optimizer + ' ' + str(Batch_size) + ' ver'

        self.OHE = True
        self.NeedPredictAfterTrain = False

    def Fit(self, X, Y, validation_data, Remake = False):
        if Remake or self.ML_NN is None or not (self.XShape == X.shape).all() or not (self.YShape == Y.shape).all():
            self.XShape = np.array(X.shape)
            self.YShape = np.array(Y.shape)

            random.seed(42)

            self.ML_NN = self.GetModel(X.shape, Y.shape, self.Layers)
            self.ML_NN.compile(loss = self.loss, optimizer=self.optimizer, metrics=self.metrics)

            self.ML_NN.summary()

            self.model_checkpoint_callback = ModelCheckpoint(
                filepath=f'{self.ModelsPath}{self.Report()}.h5',
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.00002)
            self.FitParams['callbacks'] = [self.model_checkpoint_callback]#, self.reduce_lr]

        Shape = list(self.XShape)
        Shape.append(1)
        History = self.ML_NN.fit(X.reshape(Shape), Y, **self.FitParams, validation_data = validation_data)
        self.ML_NN = load_model(f'{self.ModelsPath}{self.Report()}.h5')
        if self.verbose > 0:
            plt.figure(figsize=(14, 7))

            if 'val_accuracy' in History.history:
                plt.plot(History.history['val_accuracy'],
                     label='Доля верных ответов на тестовом наборе')
            plt.plot(History.history['accuracy'],
                     label='Доля верных ответов на тренировочном наборе')
            plt.xlabel('Эпоха обучения')
            plt.ylabel('Доля верных ответов')
            plt.legend()
            plt.show()

    def GetModel(self, X, Y, Layers):
        return None

class LSTMModel(TNN):
    def __init__(self, LSTMLayers = [100], DenseLayers = [500], Epochs=20,
                 Batch_size=64, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        TNN.__init__(self, Epochs=Epochs, Layers={'LSTM':LSTMLayers, 'D':DenseLayers},
                     Batch_size=Batch_size, loss=loss, optimizer=optimizer, metrics=metrics)

    def GetModel(self, XShape, YShape, Layers):
        LSTMLayers = Layers['LSTM']
        DenseLayers = Layers['D']

        Res = 'LSTM' + str(LSTMLayers[0])
        model = Sequential()
        #model.add(LSTM(LSTMLayers[0], input_shape=(XShape[-1], 1), return_sequences=True))#len(LSTMLayers) > 1))
        model.add(LSTM(LSTMLayers[0], input_shape=(XShape[-1], 1), return_sequences=len(LSTMLayers) > 1))

        for n in LSTMLayers[1:]:
            model.add(LSTM(n))

            Res += '_' + str(n)

        model.add(Dense(DenseLayers[0], activation="relu"))
        Res += ' D'+ str(DenseLayers[0])
        for n in DenseLayers[1:]:
            model.add(Dense(n, activation="relu"))
            Res += '_' + str(n)

        model.add(Dense(YShape[-1], activation="softmax"))
        self. LayersReport = Res

        return model

    def Report(self): # вернет строку для имени файла результата
        return self.LayersReport + ' ' + self.Descr

class LSTMExModel(TNN):
    def __init__(self, LSTMLayers = [100], DenseLayers = [500], Epochs=20,
                 Batch_size=64, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], Variant = 0, UseGRU = True):
        TNN.__init__(self, Epochs=Epochs, Layers={'LSTM':LSTMLayers, 'D':DenseLayers},
                     Batch_size=Batch_size, loss=loss, optimizer=optimizer, metrics=metrics)
        self.Variant = Variant
        self.Descr += str(self.Variant)
        self.UseGRU = UseGRU

    def GetModel(self, XShape, YShape, Layers):
        LSTMLayers = Layers['LSTM']
        DenseLayers = Layers['D']

        Res = ('LSTM' if not self.UseGRU else 'GRU') + str(self.Variant) + '_' + str(LSTMLayers[0])

        if self.Variant < 10:
            model = Sequential()

            if self.UseGRU:
                model.add(GRU(LSTMLayers[0], input_shape=(XShape[-1], 1), return_sequences=True))  # len(LSTMLayers) > 1))
            else:
                model.add(LSTM(LSTMLayers[0], input_shape=(XShape[-1], 1), return_sequences=True))#len(LSTMLayers) > 1))

            #model.add(LSTM(LSTMLayers[0], input_shape=(XShape[-1], 1), return_sequences=len(LSTMLayers) > 1))
            if self.Variant == 1 or self.Variant == 3 or self.Variant == 5 or self.Variant == 7:
                model.add(BatchNormalization())
            for n in LSTMLayers[1:]:
                model.add(LSTM(n, return_sequences=True) if not self.UseGRU else GRU(n, return_sequences=True))
                if self.Variant == 1 or self.Variant == 3:
                    model.add(BatchNormalization())
                Res += '_' + str(n)

            model.add(Flatten())

            Res += ' D' + str(DenseLayers[0])
            model.add(Dense(DenseLayers[0], activation="relu"))
            if self.Variant == 2 or self.Variant == 3 or self.Variant == 7:
                model.add(Dropout(0.2))
            for n in DenseLayers[1:]:
                model.add(Dense(n, activation="relu"))
                if self.Variant == 2 or self.Variant == 3:
                    model.add(Dropout(0.2))
                Res += '_' + str(n)

            model.add(Dense(YShape[-1], activation="softmax"))
        elif self.Variant < 20:
            Inp = Input(shape=(XShape[-1], 1))

            if self.UseGRU:
                XLSTM = GRU(LSTMLayers[0], input_shape=(XShape[-1], 1), return_sequences=len(LSTMLayers) > 1)(Inp)
            else:
                XLSTM = LSTM(LSTMLayers[0], input_shape=(XShape[-1], 1), return_sequences=len(LSTMLayers) > 1)(Inp)
            for i, n in enumerate(LSTMLayers[1:]):

                if self.UseGRU:
                    XLSTM = GRU(n, return_sequences=i < len(LSTMLayers) - 2)(XLSTM)
                else:
                    XLSTM = LSTM(n, return_sequences=i < len(LSTMLayers) - 2)(XLSTM)

                if self.Variant == 11 or self.Variant == 13:
                    XLSTM = BatchNormalization()(XLSTM)

                Res += '_' + str(n)

            InpDense = Reshape((XShape[-1],))(Inp)
            XDENSE = concatenate([XLSTM, InpDense])

            Res += ' D' + str(DenseLayers[0])
            XDENSE = Dense(DenseLayers[0], activation="relu")(XDENSE)
            if self.Variant == 12 or self.Variant == 13 or self.Variant == 17:
                XDENSE = Dropout(0.2)(XDENSE)

            for n in DenseLayers[1:]:
                XDENSE = Dense(n, activation="relu")(XDENSE)
                if self.Variant == 22 or self.Variant == 23:
                    XDENSE = Dropout(0.2)(XDENSE)
                Res += '_' + str(n)

            XDENSE = Dense(YShape[-1], activation="softmax")(XDENSE)

            model = Model(inputs=Inp, outputs=XDENSE)
        else:
            Inp = Input(shape=(XShape[-1], 1))

            if self.UseGRU:
                XLSTM = GRU(LSTMLayers[0], input_shape=(XShape[-1], 1), return_sequences=True)(Inp)
            else:
                XLSTM = LSTM(LSTMLayers[0], input_shape=(XShape[-1], 1), return_sequences=True)(Inp)#=len(LSTMLayers) > 1)(Inp)

            for i, n in enumerate(LSTMLayers[1:]):
                if self.UseGRU:
                    XLSTM = GRU(n, return_sequences=True)(XLSTM)#i < len(LSTMLayers) - 2)(XLSTM)
                else:
                    XLSTM = LSTM(n, return_sequences=True)(XLSTM)#i < len(LSTMLayers) - 2)(XLSTM)

                if self.Variant == 21 or self.Variant == 23:
                    XLSTM = BatchNormalization()(XLSTM)

                Res += '_' + str(n)

            XLSTM = Flatten()(XLSTM)

            Res += ' D' + str(DenseLayers[0])
            XLSTM = Dense(DenseLayers[0], activation="relu")(XLSTM)

            InpDense = Reshape((XShape[-1],))(Inp)
            XDENSE = concatenate([XLSTM, InpDense])

            Res += ' D' + str(DenseLayers[0])
            XDENSE = Dense(DenseLayers[0], activation="relu")(XDENSE)
            if self.Variant == 22 or self.Variant == 23 or self.Variant == 27:
                XDENSE = Dropout(0.2)(XDENSE)

            for n in DenseLayers[1:]:
                XDENSE = Dense(n, activation="relu")(XDENSE)
                if self.Variant == 22 or self.Variant == 23:
                    XDENSE = Dropout(0.2)(XDENSE)
                Res += '_' + str(n)

            XDENSE = Dense(YShape[-1], activation="softmax")(XDENSE)

            model = Model(inputs=Inp, outputs=XDENSE)

        self. LayersReport = Res

        return model

    def Report(self): # вернет строку для имени файла результата
        return self.LayersReport + ' ' + self.Descr



class DenseModel(TNN):
    def __init__(self, Layers = [500], Epochs=20,
                 Batch_size=64, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], Variant = 0):
        TNN.__init__(self, Epochs=Epochs, Layers=Layers,
                     Batch_size=Batch_size, loss=loss, optimizer=optimizer, metrics=metrics)
        self.Variant = Variant
        #self.Descr += str(self.Variant)

    def GetModel(self, XShape, YShape, Layers):
        Res = 'D' + str(self.Variant) + '_' + str(Layers[0])

        if True: #self.Variant < 10:
            Res += ' D' + str(Layers[0])
            model = Sequential()
            model.add(Dense(Layers[0], activation="relu", input_shape=(XShape[-1],)))
            if self.Variant == 2:
                model.add(Dropout(0.3))
            elif self.Variant == 1:
                model.add(BatchNormalization())

            for n in Layers[1:]:
                model.add(Dense(n, activation="relu"))
                if self.Variant == 2:
                    model.add(Dropout(0.3))
                elif self.Variant == 1:
                    model.add(BatchNormalization())
                Res += '_' + str(n)

            model.add(Dense(YShape[-1], activation="softmax"))

        #else:
        #    return None
        self. LayersReport = Res

        return model

    def Report(self): # вернет строку для имени файла результата
        return self.LayersReport + ' ' + self.Descr

class ConvModel(TNN):
    def __init__(self, ConvLayers = [100], DenseLayers = [500], Epochs=20,
                 Batch_size=64, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], Variant = 0):
        TNN.__init__(self, Epochs=Epochs, Layers={'Conv':ConvLayers, 'D':DenseLayers},
                     Batch_size=Batch_size, loss=loss, optimizer=optimizer, metrics=metrics)
        self.Variant = Variant
        self.Descr += str(self.Variant)

    def GetModel(self, XShape, YShape, Layers):
        ConvLayers = Layers['Conv']
        DenseLayers = Layers['D']

        Res = 'Conv' + str(self.Variant) + '_' + str(ConvLayers[0])

        #if self.Variant < 10:
        Inp = Input(shape=(XShape[-1], 1))
        XC = Conv1D(ConvLayers[0], 7, activation="relu", padding = "same")(Inp)

        for i, n in enumerate(ConvLayers[1:]):
            if n < 0: # MaxPooling
                XC = MaxPooling1D()(XC)
            else:
                XC = Conv1D(n, 7, activation="relu", padding = "same")(XC)

        XC = Flatten()(XC)

        Res += ' D' + str(DenseLayers[0])
        XC = Dense(DenseLayers[0], activation="relu")(XC)
        if self.Variant == 2:
            model.add(Dropout(0.3))
        elif self.Variant == 1:
            model.add(BatchNormalization())

        for n in DenseLayers[1:]:
            XC = Dense(n, activation="relu")(XC)
            if self.Variant == 2:
                model.add(Dropout(0.3))
            elif self.Variant == 1:
                model.add(BatchNormalization())
            Res += '_' + str(n)

        XC = Dense(YShape[-1], activation="softmax")(XC)

        model = Model(inputs=Inp, outputs=XC)


        self. LayersReport = Res

        return model

    def Report(self): # вернет строку для имени файла результата
        return self.LayersReport + ' ' + self.Descr


#class

'''
def DenseNN(TML_NN):
    Layers = (1000, 150)):
    modelD = Sequential()

    modelD.add(Dense(Layers[0], input_shape=70, activation="relu"))
    for n in Layers[1:]:
         modelD.add(Dense(n, activation="relu"))

    modelD.add(Dense(7, activation="softmax"))

    #Компилируем
    modelD.compile(loss="mse", optimizer=Adam(lr=1e-4))
'''

if __name__ == '__main__':
    a = 0