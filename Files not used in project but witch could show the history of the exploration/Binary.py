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
from tqdm.auto import tqdm
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import classification_report

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')



print('Reading Dataset')
X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
XTest, YTest = ReadCsv(CrPath, DelZeros = True, SortCtg = True, Train = False, Au = True)
Data = (X, Y, XTest, YTest)
Y = Y[:,1]

X0 = X[Y==0]
X0Test = X0[-50:]
X0 = X0[:-50]
X1 = X[Y!=0]
X1Test = X1[-50:]
X1 = X1[:-50]

X0, Y0 = TimeAugmentation(X0, np.zeros( (len(X0), 1) ), K=1+len(X1)*15//len(X0), random=1, LevelK=0.2)
X1, Y1 = TimeAugmentation(X1, np.ones( (len(X1), 1) ), K=15, random=1, LevelK=0.2)

X = np.concatenate([X0, X1])
Y = np.concatenate([Y0, Y1])

XTest = np.concatenate([X0Test, X1Test])
YTest = np.concatenate([np.zeros( (len(X0Test), 1) ), np.ones( (len(X1Test), 1) )])


Ind = np.arange(len(X))

np.random.shuffle(Ind)
X = X[Ind]
Y = Y[Ind]

TestForest(X, Y, XTest, YTest)

CrModel =  LSTMExModel(LSTMLayers = [10], DenseLayers = [10], Epochs=1, Batch_size=128, Variant = 23, UseGRU=True)
Filter = 0

SimpleExp = TExperiment(CrModel, Filter = Filter, InAugmentation = 0, OutAugmentation = 1, KAu = 33, InAuLevelK = 0.2, OutAuLevelK = 0.2)
SimpleExp.Execute(Data, GenCsv = True, DoubleFit = False, SinMode = False)

#FurieExp = TryFurie(CrModel, Filter = Filter, InAugmentation = 1, OutAugmentation = 0)
#FurieExp.Execute(Data, GenCsv = True)

#PolyExp = TryPoly(CrModel, Filter = Filter, deg = 35, InAugmentation = 1)
#PolyExp.Execute(Data, GenCsv = True)


List0 = SimpleExp.MultiExecute(self, Data, 11)

Ansamble = TAnsamble( (List) )

a = 0
