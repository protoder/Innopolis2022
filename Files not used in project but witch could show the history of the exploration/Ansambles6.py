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
from Libs import *
from XGBoostFeature import *

Folder = fr'{CrPath}/Models/'

# Точное воспроизводство хромосомы - при значениях по умолчанию
def ReadHromosoms(Folder, Verbose = 1):
    _x, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
    y_valid = Y[:100]
    y_train = Y[100:]

    list0 = glob.glob(Folder + "XGB3*_0.95*.npy")

    GPU = is_gpu_available()
    ModelsReiting = []
    K = 20
    y_train = np.tile(y_train, [K, 1])

    XValid = np.load(f'{CrPath}Input/Valid_100.npy')
    X = np.load(f'{CrPath}Input/Train_100.npy')

    ResX0, YTest = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)
    Ftr_Test = np.load(f'{CrPath}Input/Ftr_test.npy')
    XTest = np.concatenate((Ftr_Test, ResX0), -1)

    y_valid = np.tile(y_valid, [K, 1])

    fInd = np.load(f'{CrPath}Input/FeatureIndex0.npy')
    XTest = XTest[:, fInd]

    #Files = [list0[i] for i in Ind]

    IsFirst = True
    for i, HrFile in enumerate(list0):
        Hr = np.load(HrFile)
        #Hr[10] = 0
        test_pred, accuracy, y_pred, y_pred_cat = BoostHr(Hr,
                                                          X0=X, Y0=y_train[:, 1:2],
                                                          XValid=XValid, YValid = y_valid[:, 1:2],
                                                          X1=X, Y1=y_train[:, 1:2],
                                                          XTest= XTest,
                                                          SeedRandom=True, GPU=GPU, UseMetric="mlogloss", Verbose=Verbose)

        WriteCsv(HrFile.replace('.npy', f'ans_hist_{accuracy:0.4f}.csv' if GPU else f'ans_{accuracy:0.4f}.csv'), YTest[:, 1:2], y_pred)

        if IsFirst:
            TrainRes = np.zeros((len(list0), len(test_pred)))
            TestRes  = np.zeros((len(list0), len(y_pred)))


        TrainRes[i: i+1] = test_pred
        TestRes[i: i+1] = y_pred

        Reiting = float(HrFile[-10:-4])
        ModelsReiting.append(Reiting)

        MRName = 'SmpAnsambles_reitings.npy'
        TrainResName = 'SmpTrains.npy'
        TestResName = 'SmpTests.npy'
        FilesName = 'SmpFiles.npy'

        np.save(MRName, ModelsReiting)
        np.save(TrainResName, TrainRes)
        np.save(TestResName, TestRes)

    return ModelsReiting, TrainRes, TestRes, list0
    # Загрузили список хромосом
random.seed(25)

ReadHromosoms('e:/Data/')
