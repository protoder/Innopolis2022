import numpy as np
import random
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import os
import os.path
from tqdm.auto import tqdm
import glob
import warnings
from tensorflow.test import is_gpu_available

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

    sys.path.append('/content/drive/MyDrive/Uinnopolis')
else:
    Acer = not os.path.exists("E:/Uinnopolis/")
    CrPath = "C:/Uinnopolis/" if Acer else "E:/Uinnopolis/"

import os, glob

from Libs import *  # ReadCsv, WriteCsv, Graphic, Filter

from BaseGenetic import *

ModelsPath = 'NNModels/'


def XGBoostHromosom(Hr, X, Y, XTest, WinSize = 500, Verbose=1, GPU = 0, UseMetric="mlogloss"):
    Start = 500#int(Hr[12] * (len(X) - WinSize))

    X1, Y1 = TimeAugmentation(np.delete(X, Start + np.arange(WinSize), axis=0),
                              np.delete(Y, Start + np.arange(WinSize), axis=0),
                              K=1 + int(Hr[1] * 20), random=int(Hr[4] * 65535),
                              LevelK=0.01 + Hr[2] / 3.3, UseMain= True)#Hr[13] > 0.9)

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
                          n_estimators= 50, #int(Hr[11] * 10000),
                          max_depth=int(Hr[4] * 7) + 3,
                          min_child_weight=int(Hr[5] * 5) + 1,
                          max_bin=int(Hr[0] * 100) + 5,
                          gamma=Hr[6] / 2,
                          subsample=Hr[7] * 0.4 + 0.6,
                          colsample_bytree=Hr[8] * 0.4 + 0.6,
                          reg_alpha=RA[int(Hr[9] * 14)],  # 0.005,
                          objective='binary:logistic',
                          nthread=6,
                          scale_pos_weight=1,
                          seed=int(Hr[3] * 65535),
                          tree_method=CrMethod,
                          random_state=int(Hr[3] * 65535),
                          verbose=Verbose
                          ,silent=True)

    es = EarlyStopping(
        rounds=15,
        save_best=True,
        maximize=False,
        data_name="validation_0",
        metric_name=UseMetric
    )

    eval_set = [(ValidX, ValidY)]

    model.fit(X1, Y1, eval_metric=UseMetric, eval_set=eval_set,
              verbose=Verbose > 0, callbacks=[es])

    # make predictions for test data
    y_pred = model.predict(ValidX)

    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(ValidY, predictions)
    print('Accuracy: %.2f%%' % (accuracy * 100.0))

    if XTest is not None and accuracy > 0.93:
        y_pred = model.predict(XTest)
        return accuracy, y_pred
    else:
        return accuracy, None


class TXGBGenetic(TBaseGenetic):  # TDistributedGenetic):
    def __init__(self, X, Y, XTest, YTest, Seed=None, StopFlag=2, TreeVerbose=0):
        TBaseGenetic.__init__(self, HromosomLen=12 + 2, FixedGroupsLeft=0,
                              StopFlag=StopFlag, TheBestListSize=50,
                              StartPopulationSize=25, PopulationSize=50)
        np.random.seed(Seed)
        self.X, self.Y = X, Y
        self.XTest = XTest
        self.YTest = YTest[:, 1:2]
        self.InverseMetric = False  # т.е. опимизация на возрастание
        self.Metric = 0
        self.StoredPath = f'{CrPath}copy/XGB1copy.dat'
        self.TreeVerbose = TreeVerbose

    def TestHromosom(self, Hr, Id):
        if True:  # not Debug:
            Hr[0]
            CrValue, Test = XGBoostHromosom(Hr, self.X, self.Y, self.XTest, GPU=self.GPU, Verbose=self.TreeVerbose)

            if Test is not None:
                #WriteCsv(fr'{CrPath}XGB{int(Id)}_{CrValue:.4f}.csv', self.YTest, Test)
                np.save(fr'{CrPath}XGB{int(Id)}_{CrValue:.4f}.npy', Hr)
        else:
            CrValue = 1 - random.random() / 100
        return CrValue

    def GenerateHromosom(self, GetNewID=True):
        return np.random.random(size=self.HromosomLen)

    def Start(self):
        self.GPU = is_gpu_available()
        super().Start()

random.seed(25)

X, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True)
X0, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
ResX0, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True)
ResX1, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)
# Y = Y[:, 1]
# ResY = ResY[:, 1]


if True:
    Gn = TXGBGenetic(X, Y, ResX0, ResY, StopFlag=0, TreeVerbose = 0)
    Gn.PDeath = 0.5
    Gn.PMutation = 0.5  # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
    Gn.PMultiMutation = 0.3
    Gn.PCrossingover = 0.5
    Gn.Start()
else:
    GPU = is_gpu_available()

    Hr = np.ones(28) * 0.5
    Hr[11] = 1
    Hr[4] = 0.14285714285714285714285714285714
    Hr[5] = 1
    Hr[6] = 0
    Hr[9] = 0.35714285714285714285714285714286
    Hr[10] = 0.15

    Acc, y_pred = XGBoostHromosom(Hr, X, Y, ResX0, GPU)

    print('Acc: %.2f%%' % (Acc * 100.0))
    WriteCsv(fr'{CrPath}XGB.csv', ResY[:, 1:2], y_pred)