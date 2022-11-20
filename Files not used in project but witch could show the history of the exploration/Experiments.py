import math

import numpy as np
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model  # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, SpatialDropout2D, MaxPooling2D, \
    AveragePooling2D, Conv2D, BatchNormalization  # Импортируем стандартные слои keras
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

from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import classification_report

from Libs import *

# отладочная процедура. Берт аугментацию из внешнего файла

def ExTimeAugmentation(X, Y, K, Ver = 0, random = None, LevelK=0.05, Train = False, SinMode = True):
    TFlag = '' if Train else 'Full'

    if SinMode:
        SFlag = 's'
    else:
        SFlag = ''

    IsTrain = len(X) > 4000
    if SinMode:
        M = np.arange(X.shape[-1]) + 1
        P = np.random.random(K)

        PiK = 35 / math.pi
        NewX = [X * (1 - LevelK * (np.sin(M * PiK * P[i]))) for i in range(K - 1)]

    else:
        NewX = [X * (np.random.random_sample(size=X.shape) * LevelK + (1 - LevelK)) for i in range(K-1)]

    NewX.append(X)

    X = np.concatenate(NewX)
    Y = np.tile(Y, [K, 1])

    return X, Y

class TML_NN():
    def __init__(self, ML_NN, FitParams = {}, PredictParams = {}):
        self.ML_NN = ML_NN
        self.FitParams = FitParams
        self.PredictParams = PredictParams
        self.OHE = False
        self.NeedPredictAfterTrain = True


    def Fit(self, X, Y, validation_data, Remake = False):
        self.ML_NN.fit(X, Y, *self.FitParams)

    def Predict(self, X):
        return self.ML_NN.predict(X, *self.PredictParams)

    def Report(self): # вернет строку для имени файла результата
        return ''

class TRandomForestClassifier(TML_NN):
    def __init__(self, n_estimators = 120):
        TML_NN.__init__(self, RandomForestClassifier(random_state=0, n_estimators=n_estimators))
        self.n_estimators = n_estimators

    def Fit(self, X, Y, validation_data, Remake = False):
        if Remake:
            self.ML_NN = RandomForestClassifier(random_state=0, n_estimators=self.n_estimators)

        self.ML_NN.fit(X, Y, *self.FitParams)

    def Report(self):
        return f'RndFrst{self.n_estimators}_'

class TExperiment():
    def __init__(self, ML_NN, Filter = 0, TestFilter = 0, OutAugmentation = 0, InAugmentation = 0,
                       InAuLevelK = 0.1, OutAuLevelK = 0.1, KAu = 10, DoubleFit = True, Percentiles = []):
        self.Prefix = self.GetPrefix()
        self.ML_NN = ML_NN
        self.KFilter = Filter
        self.KTestFilter = TestFilter if TestFilter > 0 else Filter
        self.OutAugmentation = OutAugmentation
        self.InAugmentation = InAugmentation
        self.InAuLevelK = InAuLevelK
        self.OutAuLevelK = OutAuLevelK
        self.KAu = KAu

        self.OHE = ML_NN.OHE
        self.Percentiles = Percentiles

    def Execute(self, Data, GenCsv = True, Predict = True, DrawRes = False, DrawTest = False, Ver = 0, DoubleFit = True, ReportPath = './', SinMode = True):
        print()
        print('Start ' + self.Report())
        X, Y, XTest, YTest = Data

        if self.InAugmentation == 1:
            XFull, YFull = TimeAugmentation(X, Y, K = self.KAu, random = 0, LevelK = self.InAuLevelK, Ver = Ver, Train = False, SinMode = SinMode)
            X, Y = TimeAugmentation(X, Y, K=self.KAu, random=0, LevelK=self.InAuLevelK, Ver=Ver, Train=True, SinMode = SinMode)
        else:
            XFull = X
        if self.KFilter != 0:
            X = Filter(X, 1/self.KFilter)
            XFull = Filter(XFull, 1 / self.KFilter)

        if self.InAugmentation == 2:
            XFull, YFull = TimeAugmentation(X, Y, K = self.KAu, random = 0, LevelK = self.InAuLevelK, Ver = Ver, Train = False, SinMode = SinMode)
            X, Y = TimeAugmentation(X, Y, K = self.KAu, random = 0, LevelK = self.InAuLevelK, Ver = Ver, Train = True, SinMode = SinMode)

        Res = self.Transpose(X)
        XFull = self.Transpose(XFull)

        if isinstance(Res, list) or isinstance(Res, tuple):
            ServRes = Res[1]
            Res = Res[0]
            XFull = XFull[0]
        else:
            ServRes = Res

        if self.InAugmentation == 3:
            XFull, YFull = TimeAugmentation(X, Y, K = self.KAu, random = 0, LevelK = self.InAuLevelK, Ver = Ver, Train = False, SinMode = SinMode)
            Res, Y = TimeAugmentation(Res, Y, K = self.KAu, random = 0, LevelK = self.InAuLevelK, Ver = Ver, Train = True, SinMode = SinMode)

        if DrawRes:
            ServRes = self.Inverse(ServRes)

            if ServRes is not None:
                Graphic(ServRes, Label=None, Cnt=10, Together=False, File=f'{ReportPath}{self.Prefix}Graph')

        #X_train, X_test, y_train, y_test = train_test_split(Res, Y[:, 1], test_size=0.95, random_state=42)

        StartTest = -500#(len(Res) // 10)

        if self.InAugmentation == 0:
            X_train = Res[:StartTest]
            X_test = Res[StartTest:]
            y_train = Y[:StartTest, 1]
            y_test = Y[StartTest:, 1:2]

        else:
            X_train = Res
            X_test = Data[0][StartTest:]
            y_train = Y[:, 1]
            y_test = Data[1][StartTest:, 1:2]

        X_test, y_test = TimeAugmentation(X_test, y_test, K=3, random=1, LevelK=self.OutAuLevelK, Ver=Ver,
                                              SinMode=SinMode)

        print('fit train')

        if self.OHE:
            validation_data = (X_test, to_categorical(y_test))

            self.ML_NN.Fit(X_train, to_categorical(y_train), validation_data = validation_data)
        else:
            validation_data = (X_test, y_test)
            self.ML_NN.Fit(X_train, y_train, validation_data = validation_data)

        if self.ML_NN.NeedPredictAfterTrain:
            print('Оценка точности')
            pred = self.ML_NN.Predict(X_test)

            if self.OHE:
                pred = np.argmax(pred, axis=-1)

            print(self.Report(), self.ML_NN.ML_NN, '\n', recall_score(y_test, pred, average="macro", zero_division=0))
            print(0, classification_report(pred, y_test))
        if GenCsv or Predict:
            # Для повышения результата перед отправкой на проверку обучаем модель на полном датасете, без тестирования
            print('fit test')

            YTestLen = len(YTest)

            if self.OutAugmentation == 1:
                XTest, YTest = TimeAugmentation(XTest, YTest, K=self.KAu, random=1, LevelK=self.OutAuLevelK, Ver = Ver, SinMode = SinMode)

            if self.KTestFilter != 0:
                XTest = Filter(XTest, 1 / self.KTestFilter)

            if self.OutAugmentation == 2:
                XTest, YTest = TimeAugmentation(XTest, YTest, K=self.KAu, random=1, LevelK=self.OutAuLevelK, Ver = Ver, SinMode = SinMode)

            if DoubleFit:
                if self.OHE:
                    y_train_ = to_categorical(YFull[:, 1])
                    y_train = self.ML_NN.Fit(XFull, y_train_, validation_data = None, Remake = True)
                else:
                    self.ML_NN.Fit(X = XFull, Y = YFull[:, 1], validation_data = None, Remake = True)

            print('Готовлю вывод')
            Res = self.Transpose(XTest)

            if self.OutAugmentation == 3:
                Res, YTest = TimeAugmentation(XTest, YTest, K=self.KAu, random=1, LevelK=self.OutAuLevelK, Ver = Ver, SinMode = SinMode)


            if isinstance(Res, list) or isinstance(Res, tuple):
                ServRes = Res[1]
                Res = Res[0]
            else:
                ServRes = Res

            if DrawTest:
                ServRes = self.Inverse(ServRes)

                if ServRes is not None:
                    Graphic(ServRes, Label=None, Cnt=10, Together=False, File=f'{ReportPath}{self.Prefix}Test')

            predict = self.ML_NN.Predict(Res)

            if self.OHE:
                predict = np.argmax(predict, axis=-1)

            if self.OutAugmentation > 0:
                    predict = np.reshape(predict, (1, self.KAu, YTestLen, 1))
                    Res = [0]*7
                    Res[0] = (predict == 0)
                    Res[1] = (predict == 1)
                    Res[2] = (predict == 2)
                    Res[3] = (predict == 3)
                    Res[4] = (predict == 4)
                    Res[5] = (predict == 5)
                    Res[6] = (predict == 6)
                    R = np.concatenate(Res)
                    #R = R.sum(axis = 0)
                    predict = np.argmax(R.T.sum(-2), axis=-1).reshape((R.shape[2], 1))

            WriteCsv(fr'{ReportPath}{self.ML_NN.Report()}{self.Report()}.csv', YTest[:YTestLen, 1:2], predict)

            return predict
        else:
            return None

    def MultiExecute(self, Data, Cnt):
       Res = [0]*Cnt
       for i in tqdm(range(Cnt)):
           Res[i] = self.Execute(self, Data, GenCsv=False, Predict=True, DrawRes=False, DrawTest=False, Ver = i)

       return Res

    # методы для перекрытия
    # Преобразование. Если преобразование, необходимое для инвертирования, отлично от преобразования для predict,
    # вернет PredictRes, InverseRes
    def Transpose(self, X):
        return X

    # Вернет None если невозможно обратное преобразование.
    def Inverse(self, X):
        return X

    def Report(self):
        if self.KFilter == 0:
            Res = self.GetPrefix()
        else:
            Res = self.GetPrefix() + ' filter' + str(self.KFilter)

        if self.InAugmentation > 0:
            Res = Res + f' Au{self.InAugmentation}_{self.InAuLevelK}_{self.KAu}'

        if self.OutAugmentation > 0:
            Res = Res + f' OutAu{self.OutAugmentation}_{self.OutAuLevelK}_{self.KAu}'

        return Res

    def GetPrefix(self):
        return ''



class TryFurie(TExperiment):
    def __init__(self, ML_NN, Filter=0, TestFilter=0, Frqs=22, OutAugmentation=0, InAugmentation=0,
                 InAuLevelK=0.1, OutAuLevelK=0.1, KAu = 10):
        TExperiment.__init__(self, ML_NN=ML_NN, Filter=Filter, TestFilter=TestFilter, OutAugmentation=OutAugmentation,
                             InAugmentation=InAugmentation, InAuLevelK=InAuLevelK, OutAuLevelK=OutAuLevelK, KAu = KAu)

        self.Frqs = Frqs

    def Transpose(self, X):
        Res = np.fft.rfft(X)
        #DrawRes = np.fft.rfft(X[:20])
        return np.real(Res[:, :self.Frqs]), Res

    def Inverse(self, X):
        return np.fft.irfft(X)

    def GetPrefix(self):
        return 'TTF'

#class TryMainPlusFurie(TryFurie):

class TryPoly(TExperiment):
    def __init__(self, ML_NN, Filter = 0, TestFilter = 0, deg = 8, OutAugmentation = 0, InAugmentation = 0,
                       InAuLevelK = 0.1, OutAuLevelK = 0.1, KAu = 10):
        TExperiment.__init__(self, ML_NN = ML_NN, Filter = Filter, TestFilter = TestFilter, OutAugmentation = OutAugmentation,
                             InAugmentation = InAugmentation, InAuLevelK = InAuLevelK, OutAuLevelK = OutAuLevelK, KAu = KAu)

        self.deg = deg

    def Transpose(self, X):
        return np.polyfit(range(70), X.T, deg=self.deg).T

    def Inverse(self, X):
        Values = np.tile(np.arange(70), [len(X), 1]).T
        return np.polyval(X.T, Values).T

    def GetPrefix(self):
        return 'Poly'


'''
def TryFurie(X, Y, DrawRes = True, Prefix = 'FFT'):
    Poly = np.fft.rfft(X)

    if DrawRes:
        Res = np.fft.irfft(Poly)
        Graphic(Res, Label=None, Cnt=10, Together=False, File=f'{ReportPath}{Prefix}Graph')

    X_train, X_test, y_train, y_test = train_test_split(X, Y[:, 1], test_size=0.3, random_state=42)

    return np.real(Poly)

def TryPoly(X, Y, DrawRes = True, Prefix = 'P'):
    Poly = np.polyfit(range(70), X.T, deg=self.deg).T

    if DrawRes:
        Res = np.polyval(Poly.T, FX.T).T
        Graphic(Res, Label = None, Cnt=10, Together = False, File = f'{ReportPath}{Prefix}Graph')

    return np.real(Poly)
'''