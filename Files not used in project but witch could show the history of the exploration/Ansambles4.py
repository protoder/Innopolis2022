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
    X, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True)
    X0, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
    ResX0, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)

    X_valid, y_valid = X0[:100], Y[:100]
    X_train, y_train = X0[100:], Y[100:]

    FtrTrain = np.load('Ftr_train_100.npy')
    XTrain = np.concatenate([FtrTrain, X_train], -1)

    FtrFull = np.load('Ftr_train_Full.npy')
    XFull = np.concatenate([FtrFull, X0], -1)

    FtrValid = np.load('Ftr_valid_100.npy')
    XValid = np.concatenate([FtrValid, X_valid], -1)

    FtrTest = np.load('Ftr_test.npy')
    XTest = np.concatenate([FtrTest, ResX0], -1)

    fInd = np.load('FeatureIndex0.npy')

    list0 = glob.glob(Folder + "XGB3*_0.95*.npy")

    GPU = is_gpu_available()
    ModelsReiting = []

    TrainRes = None
    TestRes = None

    #Files = [list0[i] for i in Ind]

    IsFirst = True
    for i, HrFile in enumerate(list0):
        Hr = np.load(HrFile)
        #Hr[10] = 0
        test_pred, accuracy, y_pred, y_pred_cat = BoostHr(Hr,
                                                          X0=XTrain[:, fInd], Y0=y_train[:, 1:2],
                                                          XValid=XValid[:, fInd], YValid = y_valid[:, 1:2],
                                                          X1=XFull[:, fInd], Y1=Y[:, 1:2],
                                                          XTest= XTest[:, fInd],
                                                          SeedRandom=True, GPU=GPU, UseMetric="mlogloss", Verbose=Verbose)

        WriteCsv(HrFile.replace('.npy', f'_ftr_hist_{accuracy:0.4f}.csv' if GPU else f'_ftr_{accuracy:0.4f}.csv'), ResY[:, 1:2], y_pred)

        if IsFirst:
            TrainRes = np.zeros((len(list0), len(test_pred)))
            TestRes  = np.zeros((len(list0), len(y_pred)))


        TrainRes[i: i+1] = test_pred
        TestRes[i: i+1] = y_pred

        Reiting = float(HrFile[-10:-4])
        ModelsReiting.append(Reiting)

        MRName = 'Ansambles_reitings.npy'
        TrainResName = 'Trains.npy'
        TestResName = 'Tests.npy'
        FilesName = 'Files.npy'

        np.save(MRName, ModelsReiting)
        np.save(TrainResName, TrainRes)
        np.save(TestResName, TestRes)

    return ModelsReiting, TrainRes, TestRes, list0
    # Загрузили список хромосом
random.seed(25)

ReadHromosoms('e:/res/')
