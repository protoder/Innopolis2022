import numpy as np
import random
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
import os.path
from tqdm.auto import tqdm
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.utils import to_categorical

from Libs import * #ReadCsv, WriteCsv, Graphic, Filter
from Experiments import *
from NN import *

import warnings
warnings.filterwarnings("ignore")

Colab = True
try:
    from google.colab import drive
except:
    Colab = False

if Colab:
    from google.colab import drive

    # Подключаем Google drive
    drive.mount('/content/drive')
    CrPath = "/content/drive/MyDrive/Uinnopolis/"

    import sys
    sys.path.append('/content/drive/Hacaton')
else:
    Acer = not os.path.exists("E:/Uinnopolis/")
    CrPath = "C:/Uinnopolis/" if Acer else "E:/Uinnopolis/"

import os, glob

Folder = fr'{CrPath}/Models/'

def ForestCmd(X, X0, Y):
    Y = Y[:, 1]
    Y = Y.reshape((Y.shape[0], 1))
    np.random.seed(6)
    rnd = np.random.random(7)
    CrRes, CrValue, CrTest = ForestHromosom(rnd, X, Y, XTest)
    Start = 200
    X1, Y1 = TimeAugmentation(np.delete(X, Start + np.arange(100), axis=0),
                              np.delete(Y, Start + np.arange(100), axis=0),
                              K=3, random=24, LevelK=0.17, UseMain=False)

    CrRes, CrValue, CrTest = TestForest2(X1, Y1, X[Start:Start + 100], Y[Start:Start + 100], X1, RandomState=2)
    a = 0

def ForestAnsambles(X, X0, Y, N=11, TestX=None, TestY=None, ResX0=None, ResX1=None, DoTest=True):
    Res = [0] * N
    Values = [0] * N
    Tests = [0] * N
    Rnds = [0] * (N // 2)

    for FileInd in range(1000):
        if not os.path.isfile(f'{CrPath}NFTRes{FileInd}.npy'):
            break

    Y = Y[:, 1]

    for i in range(N // 2):  # CrX, CrY, CrX0 in zip(X, Y0):
        rnd = np.random.random(14)
        #rnd = [0.1, 0.07, 0.1, 0.231, 0.002, 0.0015, 0.0006, 0.1, 0.07, 0.1, 0.231, 0.002, 0.0015, 0.0006]

        CrRes, CrValue, CrTest = ForestHromosom(rnd[:7], X, Y, ResX0 )

        '''Start = int(rnd[4] * (len(X)- 100))
        print(i, 'k', 1 + int(rnd[1] * 30), 'LevelK=', 0.01 + rnd[0] / 3.3, 'Start', Start)
        X1, Y1 = TimeAugmentation(np.delete(X, Start + np.arange(100), axis = 0),
                                  np.delete(Y, Start + np.arange(100), axis = 0),
                                  K=1 + int(rnd[1] * 30), random=24, LevelK=0.01 + rnd[0] / 3.3, UseMain=i == 0)

        CrRes, CrValue, CrTest = TestForest2(X1, Y1, X[Start:Start+100], Y[Start:Start+100], ResX0, RandomState=None)'''

        CrRes = to_categorical(CrRes)
        CrRes = CrRes.reshape((1, len(CrRes), 7))

        Res[i * 2] = CrRes  # (CrRes * CrValue)
        Values[i * 2] = CrValue.reshape((1, len(CrRes)))
        CrTest = to_categorical(CrTest)
        Tests[i * 2] = CrTest.reshape((1, len(CrTest), CrTest.shape[-1]))

        CrRes, CrValue, CrTest = ForestHromosom(rnd[7:], X0, Y, ResX1)

        CrRes = to_categorical(CrRes)
        CrRes = CrRes.reshape((1, len(CrRes), 7))

        Res[i * 2 + 1] = CrRes  # (CrRes * CrValue)
        Values[i * 2 + 1] = CrValue.reshape((1, len(CrRes)))

        CrTest = to_categorical(CrTest)
        Tests[i * 2 + 1] = CrTest.reshape((1, len(CrTest), CrTest.shape[-1]))

        Rnds[i] = np.reshape(rnd, (1, 14))

        if True:
            FullRes = np.concatenate(Res[:(i+1)*2])
            FullValues = np.concatenate(Values[:(i+1)*2])
            FullTests = np.concatenate(Tests[:(i+1)*2])
            FullRnds = np.concatenate(Rnds[:(i + 1)])
            np.save(f'{CrPath}NFTRes{FileInd}.npy', FullRes)
            np.save(f'{CrPath}NFTValue{FileInd}.npy', FullValues)
            np.save(f'{CrPath}NFTTests{FileInd}.npy', FullTests)
            np.save(f'{CrPath}NFTRnds{FileInd}.npy', FullRnds)

    FullRes = np.sum(FullRes, axis=0)
    FullRes = np.argmax(FullRes.T, axis=0)

    res = recall_score(YTest, FullRes, average="macro", zero_division=0)

    print(res)
    print(0, classification_report(FullRes, YTest))

def Predict(M, CrX, CrY, Predicts):
    pred = M.predict(CrX)

# Xlist, Ylist в формате
# ModelsList - спсок словарей формата {Model, K, Type}
# Model - путь к выгруженной модели с весами, K - рейтинг модели, Type - тип, на всякий случай
def StartAnsamble(XList, YList, KAu, ModelsList):

    X, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=True, Train=False, Au=True)
    X, y = TimeAugmentation(X, y, K=KAu, random=1, LevelK=0.1, Ver=0, SinMode=False)

    XTrain, YTrain = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, Au=True)

    Predicts = []
    Results  = []
    for M in ModelsList:

        if M['Type'] == 0:
            CrX, CrY = X, Y

        M = load_model(M['Model'])


        res = M.predict(X)#(M.predict(X) * M['K'])
        res = res.reshape(res.shape.insert(1, KAu, len(Y), 1) )
        Predicts.append(res)

    Res = np.concatenate(Predicts)

    predict = np.reshape(predict, (1, self.KAu, len(Y), 1))
    predict = np.argmax(predict.T.sum(-2), axis=-1).reshape((predict.shape[2], 1))

    WriteCsv(fr'{ReportPath}{self.ML_NN.Report()}{self.Report()}.csv', YTest[:YTestLen, 1:2], predict)

random.seed(25)

X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
X0, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True, PostProc = True)
ResX0, ResY = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = False, RetPrc = True)
ResX1, ResY = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = False, RetPrc = True, PostProc = True)

ForestAnsambles(X, X0, Y, N = 2000, ResX0 = ResX0, ResX1 = ResX1)


