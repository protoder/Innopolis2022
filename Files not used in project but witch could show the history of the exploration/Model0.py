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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

random.seed(1)

print('Reading Dataset')
X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
XTest, YTest = ReadCsv(CrPath, DelZeros = True, SortCtg = True, Train = False, Au = True)
Data = (X, Y, XTest, YTest)

ResList = []
list0 = glob.glob('e:/res/* 0.9*.csv')
R = []
for F in list0:
    CrRait = float(F[-12:-4])
    df = pd.read_csv(F)
    CrList = df['crop'].to_numpy()
    CrList = to_categorical(CrList)
    CrList = CrList.reshape((1, len(CrList), 7))
    ResList.append(CrList)
    R.append(CrRait)

R = np.array(R)

#589 == 4
#680 = 6
#1169 = 4
#1339 = 6
#1394(4462)=4
#1428 = 6
if True:
    ResList = np.concatenate(ResList)[0:7]
    WResList = np.zeros_like(ResList)

    for i in range(len(ResList)):
        WResList[i:i+1] = ResList[i] * R[i]

    SM = WResList.sum(axis = 0)
    Res = np.argmax(SM, axis = -1)

    df['crop'] = Res
    df.to_csv(r'e:/musor/avg0_7.csv', sep = ',', index = False)

Ind = np.argsort(R)[::-1]

Files = [list0[i] for i in Ind]
ResList = np.concatenate(ResList)[Ind]
WResList = np.zeros_like(ResList)
W2ResList = np.zeros_like(ResList)
R = R[Ind]

for i in range(len(ResList)):
    WResList[i:i+1] = ResList[i] * R[i]
    W2ResList[i:i+1] = ResList[i] * R[i]* R[i]

WSum = WResList.sum(axis = 0)
SM = ResList.sum(axis = 0)
D = np.argmax(WSum, axis = -1) != np.argmax(SM, axis = -1)
SM0 = SM == 0
SM24 = SM == 24
Hall = (SM0 | SM24)
Hall = 1 - Hall
Mask = Hall.sum(axis= -1)

for n in range(len(ResList)//2 + 1):
    CrInd = np.nonzero(SM == len(ResList)//2+n)[0]
    W = WSum[CrInd]
    Positions = ResList[:,CrInd]
    SMPos = SM[CrInd]
    a = 0


CrModel = TRandomForestClassifier(n_estimators = 175)
#CrModel =  LSTMExModel(LSTMLayers = [100, 70], DenseLayers = [100, 100], Epochs=20, Batch_size=128, Variant = 23)
Filter = 0
SimpleExp = TExperiment(CrModel, Filter = Filter, InAugmentation = 1, OutAugmentation = 1, KAu = 31, Percentiles = Prc, InAuLevelK = 0.25, OutAuLevelK = 0.1)
SimpleExp.Execute(Data, GenCsv = True, DoubleFit = True, SinMode = True)

#FurieExp = TryFurie(CrModel, Filter = Filter, InAugmentation = 1, OutAugmentation = 0)
#FurieExp.Execute(Data, GenCsv = True)

#PolyExp = TryPoly(CrModel, Filter = Filter, deg = 35, InAugmentation = 1)
#PolyExp.Execute(Data, GenCsv = True)


List0 = SimpleExp.MultiExecute(self, Data, 11)

Ansamble = TAnsamble( (List) )

a = 0
