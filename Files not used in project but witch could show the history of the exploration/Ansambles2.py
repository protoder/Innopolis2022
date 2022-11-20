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
def BoostHrV0(Hr, X, Y, XTest, Estimators = 50, WinSize=500, Start = 500, Verbose=1, GPU=0, UseMetric="mlogloss"):

        X1, Y1 = TimeAugmentation(np.delete(X, Start + np.arange(WinSize), axis=0),
                                  np.delete(Y, Start + np.arange(WinSize), axis=0),
                                  K=1 + int(Hr[1] * 20), random=int(Hr[4] * 65535),
                                  LevelK=0.01 + Hr[2] / 3.3, UseMain=True)  # Hr[13] > 0.9)

        Y1 = Y1[:, 1:2]

        ValidX, ValidY = TimeAugmentation(X[Start:Start + WinSize],
                                          Y[Start:Start + WinSize],
                                          K=20, random=24, LevelK=0.1, UseMain=True)

        ValidY = ValidY[:, 1:2]

        RA = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100]
        Methods = ['exact', 'approx', 'hist']
        CrMethod = Methods[int(Hr[10] * 3)]

        if GPU:  # and CrMethod == 'hist':
            CrMethod = 'gpu_hist'

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
                              verbose=Verbose
                              , silent=True)

        es = EarlyStopping(
            rounds=15,
            save_best=True,
            maximize=False,
            data_name="validation_0",
            metric_name=UseMetric
        )

        eval_set = [(ValidX, ValidY)]

        model.fit(X1, Y1, eval_metric=UseMetric, eval_set=eval_set, verbose= 0, callbacks=[es])

        # make predictions for test data
        y_pred = model.predict(ValidX)

        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(ValidY, predictions)
        print('Accuracy: %.2f%%' % (accuracy * 100.0))

        if XTest is not None:
            y_pred = model.predict(XTest)
            return accuracy, y_pred
        else:
            return accuracy, None

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
def TestModels0(Folder, Limit, X, Y, XTest, YTest):
    Models, ModelsReiting, Files = ReadHromosoms(Folder)

    GPU = is_gpu_available()

    for Hr, Reiting, F in zip(Models, ModelsReiting, Files):
        if Reiting < Limit:
            break

        print(F)

        Hr[10] = 0 # без GPU первая модель
        accuracy, y_pred = BoostHrV0(Hr, X, Y, XTest, Estimators=10000, WinSize=50, GPU = GPU)

        WriteCsv(F.replace('.npy', f'_hist_{accuracy:0.4f}.csv' if GPU else f'_{accuracy:0.4f}.csv'), YTest[:,1:2], y_pred)
        print()



def PrepareResults(Y, ResY):
    list0 = glob.glob(Folder + "*FTRes*.npy")

    N = len(list0)
    Res = [0] * N
    Res1 = [0] * N
    Values = [0] * N
    Tests = [0] * N
    Hrs = [0] * N

    i = 0
    for F in list0:
        FValue = F.replace('FTRes', 'FTValue')
        FTest = F.replace('FTRes', 'FTTests')
        Hr = F.replace('FTRes', 'FTRnds')
        CrRes = np.load(F)

        if CrRes.shape[1] == 100:
            Res[i] = CrRes
            Values[i] = np.load(FValue)
            Tests[i] = np.load(FTest)
            Hrs[i] = np.load(Hr)
            i+= 1

    #Read(0, 4)
    #Read(1, 2)
    #Read(2, 5)
    #Read(3, 3)

    '''NewRes0 = [Res[0],Res[2]]
    NewRes1 = [Res[1], Res[3]]
    NewValues0 = [Values[0], Values[2]]
    NewValues2 = [Values[2], Values[2]]
    nRes = np.concatenate(NewRes0)
    nRes1 = np.concatenate(NewRes1)
    nValues = np.concatenate(NewValues0)
'''


    nRes = np.concatenate(Res[:i])
    nValues = np.concatenate(Values[:i])
    nTests = np.concatenate(Tests[:i])
    nHr = np.concatenate(Hrs[:i])

    nTests*= np.tile(nValues, 2071*7).reshape(len(nValues),2071,7)

    ind = np.argsort(nValues.reshape(len(nValues)), axis = 0)
    nRes = nRes[ind]
    nValues = nValues[ind]
    nTests = nTests[ind]
    #nHr = nHr[ind]
    Pos = 5
    #R = nTests[-Pos:-Pos+1]
    R = nTests[-20:]
    #R = nTests[-Pos:]
    R = np.sum(R, axis = 0)
    R = nTests[-1:]
    predict = np.argmax(R, axis = -1).reshape(2071,1)

    #WriteCsv(fr'e:/musor/RF {Pos - 1}.csv', ResY[:, 1:2], predict)
    WriteCsv(fr'e:/musor/Лучший.csv', ResY[:, 1:2], predict)
#ансамбль 8 взвешенный.csv - .963326
#1-я - 0.952524
#2-я -  0.959647
#3-я - 0.960513
#4-я - 0.963326
#5-я - 0.965538
#6-z -  0.956059
    a = 0




def ForestAnsambles(X, X0, Y, N=11, TestX=None, TestY=None, ResX0=None, ResX1=None, DoTest=True):
    Y = Y[:, 1]

    '''if TestX is None:
        XTest = X[:100]
        X = X[100:]

        YTest = Y[:100]
        Y = Y[100:]

        X0Test = X0[:100]
        X0 = X0[100:]'''

    Res = [0] * N
    Values = [0] * N
    Tests = [0] * N
    Rnds = [0] * (N // 2)

    Y = Y.reshape((Y.shape[0], 1))

    for FileInd in range(1000):
        if not os.path.isfile(f'{CrPath}FTRes{FileInd}.npy'):
            break

    for i in range(N // 2):  # CrX, CrY, CrX0 in zip(X, Y0):
        rnd = np.random.random(6)
        Start = int(rnd[4] * (len(X)- 200))
        print(i, 'k', 1 + int(rnd[1] * 30), 'LevelK=', 0.01 + rnd[0] / 3.3, 'Start', Start)
        X1, Y1 = TimeAugmentation(np.delete(X, Start + np.arange(200), axis = 0),
                                  np.delete(Y, Start + np.arange(200), axis = 0),
                                  K=1 + int(rnd[1] * 30), random=24, LevelK=0.01 + rnd[0] / 3.3, UseMain=i == 0)

        CrRes, CrValue, CrTest = TestForest2(X1, Y1, X[Start:Start+200], Y[Start:Start+200], ResX0, RandomState=11)
        CrRes = to_categorical(CrRes)
        CrRes = CrRes.reshape((1, len(CrRes), 7))

        Res[i * 2] = CrRes  # (CrRes * CrValue)
        Values[i * 2] = CrValue.reshape((1, len(CrRes)))
        CrTest = to_categorical(CrTest)
        Tests[i * 2] = CrTest.reshape((1, len(CrTest), CrTest.shape[-1]))

        Start = int(rnd[5] * (len(X)- 200))
        print(i, 'Расширенных вход. k', 1 + int(rnd[3] * 30), 'LevelK=', 0.01 + rnd[2] / 3.3, 'Start', Start)
        X1, Y1 = TimeAugmentation(np.delete(X0, Start + np.arange(200), axis = 0),
                                  np.delete(Y, Start + np.arange(200), axis = 0),
                                  K=1 + int(rnd[3] * 30), random=12, LevelK=0.01 + rnd[2] / 3.3, UseMain=i == 0)
        CrRes, CrValue, CrTest = TestForest2(X1, Y1, X0[Start:Start+200], Y[Start:Start+200], ResX1, RandomState=33)
        CrRes = to_categorical(CrRes)
        CrRes = CrRes.reshape((1, len(CrRes), 7))

        Res[i * 2 + 1] = CrRes  # (CrRes * CrValue)
        Values[i * 2 + 1] = CrValue.reshape((1, len(CrRes)))

        CrTest = to_categorical(CrTest)
        Tests[i * 2 + 1] = CrTest.reshape((1, len(CrTest), CrTest.shape[-1]))

        Rnds[i] = rnd.reshape((1, 6))

        if True:
            FullRes = np.concatenate(Res[:(i+1)*2])
            FullValues = np.concatenate(Values[:(i+1)*2])
            FullTests = np.concatenate(Tests[:(i+1)*2])
            np.save(f'{CrPath}FTRes{FileInd}.npy', FullRes)
            np.save(f'{CrPath}FTValue{FileInd}.npy', FullValues)
            np.save(f'{CrPath}FTTests{FileInd}.npy', FullTests)

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

    WriteCsv(fr'{ReportPath}{self.ML_NN.Report()}{self.Report()}.csv', YTest[:, 1:2], predict)

random.seed(25)

X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
X0, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True, PostProc = True)
ResX0, ResY = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = False, RetPrc = True)
ResX1, ResY = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = False, RetPrc = True, PostProc = True)

TestModels0(f'{CrPath}res/', 0.9512, X, Y, ResX0, ResY)
TestModels0(f'{CrPath}res/', 0.9512, X0, Y, ResX1, ResY)

import matplotlib.pyplot as plt

X_0 = X[Y[:,1]==0]
X_1 = X[Y[:,1]==1]

Nhist = 15

Res = np.zeros( (len(X), 7, 70, Nhist))
Mx = np.max(X)
Mn = np.min(X)

if False: # расчет общей гистограммы
    for xR in tqdm(range(len(X))):
        for yR in range(7):
            for Step in range(70):
                Res[xR, yR, Step, :] = np.histogram(X[Y[:,1]==yR][:, Step], bins=np.arange(Nhist+1) * (Mx - Mn)/Nhist + Mn)[0]

# Расчет гистограммы по X
'''ResX = np.zeros( (len(X), 70, Nhist))
for xR in tqdm(range(len(X))):
    for Step in range(70):
        ResX[xR, Step, :] = np.histogram(X[xR, Step], bins=np.arange(Nhist + 1) * (Mx - Mn) / Nhist + Mn)[0]


np.save(f'HistX{Nhist}.npy', ResX)
ResX = np.load(f'HistX{Nhist}.npy')'''

'''plt.hist(X_0[:, 0], 10, density=True)
plt.show()

plt.hist(X_1[:, 0], 10, density=True)
plt.show()
'''





Hr = [0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.5]
#pred, res, pred1 = HGFHromosom(Hr, ResX.reshape((len(ResX), 70*Nhist)), Y[:,1], None, WinSize = 100,  MaxAyLevel = 0)
pred, res = HGFHromosom(Hr, X, Y[:,1], None, WinSize = 100,  MaxAyLevel = 33)
PrepareResults(Y, ResY)

ForestAnsambles(X, X0, Y[:,1], N = 2000, ResX0 = ResX0, ResX1 = ResX1)


