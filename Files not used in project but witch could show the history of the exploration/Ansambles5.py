import numpy as np
import random
#from tensorflow.keras.models import Model, load_model  # Импортируем модели keras: Model
#from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, SpatialDropout2D, MaxPooling2D, \
#    AveragePooling2D, Conv2D, BatchNormalization  # Импортируем стандартные слои keras
#from tensorflow.keras import backend as K  # Импортируем модуль backend keras'а
#from tensorflow.keras.optimizers import Adam  # Импортируем оптимизатор Adam
#from tensorflow.keras import \
#    utils  # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
#from keras import regularizers
#from keras.callbacks import Callback
#import tensorflow as tf
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

#from xgboost import XGBClassifier
#from xgboost.callback import EarlyStopping

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

    list0 = glob.glob(Folder + "XGB3*_0.95*.npy")

    GPU = is_gpu_available()
    ModelsReiting = []

    TrainRes = None
    TestRes = None

    #Files = [list0[i] for i in Ind]

    IsFirst = True
    for i, HrFile in enumerate(list0):
        Hr = np.load(HrFile)

        model = RandomForestClassifier(random_state=232, n_estimators=127)
        model.fit(XTrain[:, fInd], y_train[:, 1:2])

        # make predictions for test data
        y_pred = model.predict(XValid[:, fInd])

        accuracy = accuracy_score(y_valid[:, 1:2], y_pred)
        print('Accuracy 0: %.8f%%' % (accuracy * 100.0))

        y_pred = model.predict(XTest[:, fInd])

        WriteCsv(HrFile.replace('.npy', f'_ftr_forest0_{accuracy:0.4f}.csv'), ResY[:, 1:2], y_pred)

        model = RandomForestClassifier(random_state=232, n_estimators=127)
        model.fit(XFull[:, fInd], Y[:, 1:2])

        # make predictions for test data
        #y_pred = model.predict(XValid[:, fInd])

        #accuracy = accuracy_score(y_valid[:, 1:2], y_pred)
        #print('Accuracy 0: %.8f%%' % (accuracy * 100.0))

        y_pred = model.predict(XTest[:, fInd])

        WriteCsv(HrFile.replace('.npy', f'_ftr_forest_full_{accuracy:0.4f}.csv'),
                 ResY[:, 1:2], y_pred)

        if IsFirst:
            TrainRes = np.zeros((len(list0), len(test_pred)))
            TestRes  = np.zeros((len(list0), len(y_pred)))


        TrainRes[i: i+1] = TrainRes
        TestRes[i: i+1] = TestRes

        Reiting = float(HrFile[-10:-4])
        ModelsReiting.append(Reiting)

        MRName = 'Ansambles_reitings.npy'
        TrainResName = 'Trains.npy'
        TestResName = 'Tests.npy'
        FilesName = 'Files.npy'

        np.save(MRName, ModelsReiting)
        np.save(TrainResName, TrainRes)
        np.save(TestResName, TestRes)

    return ModelsReiting, TrainRes, TestRes, Files
    # Загрузили список хромосом
random.seed(25)

ReadHromosoms('e:/res/')
