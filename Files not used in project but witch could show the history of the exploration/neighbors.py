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

from Libs import * #ReadCsv, WriteCsv, Graphic, Filter
from Experiments import *
from NN import *
from sklearn.metrics import *
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import classification_report

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

random.seed(1)

print('Reading Dataset')
X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
XTest, YTest = ReadCsv(CrPath, DelZeros = True, SortCtg = True, Train = False, Au = True)
Data = (X, Y, XTest, YTest)

X_train, X_test, y_train, y_test = train_test_split(X, Y[:, 1], test_size=0.1, random_state=42)

L = Y[:,1]

CrX = X_train
CrY = y_train

CrX, CrY = TimeAugmentation(CrX, CrY, K = 10, random = None, LevelK=0.1)

TX = X_test
TY = y_test


'''
CrX = X
CrY = Y

CrX, CrY = TimeAugmentation(CrX, CrY, K = 10, random = None, LevelK=0.1)

TX = XTest
TY = YTest
'''

'''
CrX = X#X_train
CrY = Y#y_train

CrX, CrY = TimeAugmentation(CrX, CrY, K = 10, random = None, LevelK=0.1)

TX = X#X_test
TY = Y#y_test
'''


for N in range(1):

    L = CrY[:, 1]

    FirstIndexes = (9, 8, 2, 0, 1, 3, 7)
    neigh = NearestNeighbors(n_neighbors=len(CrX), metric='cosine')
    neigh.fit(CrX)
    distances, idxs = neigh.kneighbors(TX, 14 + N, return_distance=True)
    Res = L[idxs]

    Add = (0, 0, 0, 0, 0, 0, 0) # можно "подыгрывать" какому-то из результатов

    predict = np.zeros(len(Res))
    for i, Row in enumerate(zip(Res, distances)):
            Sum = [0]*7

            IsFirst = True
            BaseDist = Row[1][0]


            Sum[Row[0][0]] = BaseDist
            for CrDist, CrLabel in zip(Row[1], Row[0]):
                if IsFirst:
                    BaseDist = CrDist

                    if BaseDist == 0:
                        continue

                    Sum[CrLabel] = 1

                    IsFirst = False
                    continue

                Sum[CrLabel] += BaseDist/CrDist

            predict[i] = np.argmax(Sum)

    #WriteCsv(fr'e:/musor/nh.csv', TY[:, 1:2], predict)

    print(8+N, recall_score(TY[:,1], predict, average="macro", zero_division=0))
print(0, classification_report(predict, L))

K = [(Res[0, :i] == Test).sum() for i in range(600)]
plt.figure(figsize=(14, 7))
plt.plot(K, marker='.')
plt.show()

BestInd = np.argmax(K)[-1]
Best = (Res[0, :i] == Test).sum()
print(n, Best, BestInd)






CrModel = TRandomForestClassifier(n_estimators = 175)
#CrModel =  LSTMExModel(LSTMLayers = [100, 70], DenseLayers = [100, 100], Epochs=30, Batch_size=5000, Variant = 23)
Filter = 0
#0.9634883540096352 при 10
#0.9636136256118742 без SinMode

# 0.9898250788933062 при 30
#  0.9906881569868153 при 33 (35 меньше 30)
SimpleExp = TExperiment(CrModel, Filter = Filter, InAugmentation = 1, OutAugmentation = 1, KAu = 33, InAuLevelK = 0.1, OutAuLevelK = 0.1)
SimpleExp.Execute(Data, GenCsv = True, DoubleFit = False, SinMode = False)
#0.949556844290229
#FurieExp = TryFurie(CrModel, Filter = Filter, InAugmentation = 1, OutAugmentation = 0)
#FurieExp.Execute(Data, GenCsv = True)

#PolyExp = TryPoly(CrModel, Filter = Filter, deg = 35, InAugmentation = 1)
#PolyExp.Execute(Data, GenCsv = True)


List0 = SimpleExp.MultiExecute(self, Data, 11)

Ansamble = TAnsamble( (List) )

a = 0
