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
from tensorflow.test import is_gpu_available

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from sklearn.metrics import accuracy_score

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

# Точное воспроизводство хромосомы - при значениях по умолчанию
def BoostHrV0_1(Hr, X, Y, XTest, Estimators = 500, WinSize=500, GPU=0, UseMetric="mlogloss", Objects = 1000):
        RA = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100]
        Methods = ['exact', 'approx', 'hist']
        CrMethod = Methods[int(Hr[10] * 3)]

        if GPU:  # and CrMethod == 'hist':
            CrMethod = 'gpu_hist'

        K = 1 + int(Hr[1] * 20)
        rnd = int(Hr[4] * 65535)
        LevelK = 0.01 + Hr[2] / 3.3

        Res = [0]*Objects
        SumAcc = 0

        for i in tqdm(range(Objects)):
            Start = int(random.random() * (len(X) - WinSize))

            XMask = np.arange(len(X))
            np.random.shuffle(XMask)
            VakidMask = XMask[WinSize:]
            XMask = XMask[:WinSize]

            X1, Y1 = TimeAugmentation(X[XMask], Y[XMask],
                                      K=K, random= rnd, LevelK=LevelK, UseMain=True)  # Hr[13] > 0.9)

            ValidX, ValidY = TimeAugmentation(X[VakidMask], Y[VakidMask],K=20, random=24, LevelK=0.1, UseMain=True)

            Y1 = Y1[:, 1:2]
            ValidY = ValidY[:, 1:2]

            eval_set = [(ValidX, ValidY)]

            model = XGBClassifier(learning_rate=0.01,
                                  n_estimators=Estimators,  # int(Hr[11] * 10000),
                                  max_depth=int(Hr[4] * 7) + 3,
                                  min_child_weight=int(Hr[5] * 5) + 1,
                                  max_bin=int(Hr[0] * 100) + 5,
                                  gamma=Hr[6] / 2,
                                  subsample=Hr[7] * 0.4 + 0.6,
                                  colsample_bytree=Hr[8] * 0.4 + 0.6,
                                  reg_alpha=RA[int(Hr[9] * 14)],  # 0.005,
                                  objective='binary:logistic',
                                  nthread=8,
                                  scale_pos_weight=1,
                                  seed=int(Hr[3] * 65535),
                                  tree_method=CrMethod,
                                  random_state=int(Hr[3] * 65535),
                                  verbose=0
                                  , silent=True)

            es = EarlyStopping(
                rounds=25,
                save_best=True,
                maximize=False,
                data_name="validation_0",
                metric_name=UseMetric
            )

            model.fit(X1, Y1, eval_metric=UseMetric, eval_set=eval_set, verbose=0, callbacks=[es])

            # make predictions for test data
            y_pred = model.predict(ValidX)

            #predictions = [round(value) for value in y_pred]

            # evaluate predictions
            accuracy = accuracy_score(ValidY, y_pred)
            SumAcc+= accuracy

            print(i, 'Accuracy: %.5f%%' % (accuracy))

            if XTest is not None:
                y_pred = model.predict(XTest)
                y_pred = to_categorical(y_pred)
                y_pred = np.reshape(y_pred, (1, len(y_pred), 7))

                Res[i] = y_pred

        R = np.concatenate(Res)
        R = R.sum(axis = 0)
        ArgR =np.argmax(R, axis=-1)
        predict = ArgR.reshape((len(ArgR), 1))

        print(i, 'Итоговый Accuracy: %.5f%%' % (SumAcc))

        return SumAcc, predict, R

def ReadHromosoms(Folder):
    Models = []
    ModelsReiting = []

    list0 = glob.glob(Folder + "XGB3*_0.95*.npy")
    for F in list0:
        Hr = np.load(F)
        Hr = Hr.reshape( (1, len(Hr)))

        Reiting = float(F[-10:-4])
        Models.append(Hr)
        ModelsReiting.append(Reiting)

    Models= np.concatenate(Models)
    ModelsReiting = np.array(ModelsReiting)
    Ind = np.argsort(ModelsReiting)[::-1]
    ModelsReiting= ModelsReiting[Ind]
    Models= Models[Ind]
    Files = [list0[i] for i in Ind]

    return Models, ModelsReiting, Files
    # Загрузили список хромосом

# Простой прогон - выжимает максимум из хромосомы, создает файл с данными
def TestModels1(Folder, Limit, X, Y, XTest, YTest):
    Models, ModelsReiting, Files = ReadHromosoms(Folder)

    GPU = is_gpu_available()

    for Hr, Reiting, F in zip(Models, ModelsReiting, Files):
        if Reiting < Limit:
            break

        print(F)

        Hr[10] = 0 # без GPU первая модель
        accuracy, y_pred, Regress_prediction = BoostHrV0_1(Hr, X, Y, XTest, Estimators=10000, WinSize=int(len(X) * 0.9), GPU = GPU)

        np.save(F.replace('.npy', f'_hist_pred_{accuracy:0.4f}.npy' if GPU else f'_pred_{accuracy:0.4f}.npy'), Regress_prediction)
        WriteCsv(F.replace('.npy', f'_hist_{accuracy:0.4f}.csv' if GPU else f'_{accuracy:0.4f}.csv'), YTest[:,1:2], y_pred)
        print()

random.seed(25)

X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
X0, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True, PostProc = True)
ResX0, ResY = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = False, RetPrc = True)
ResX1, ResY = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = False, RetPrc = True, PostProc = True)

TestModels1(f'{CrPath}res/', 0.9516, X, Y, ResX0, ResY)
TestModels1(f'{CrPath}res/', 0.9516, X0, Y, ResX1, ResY)
