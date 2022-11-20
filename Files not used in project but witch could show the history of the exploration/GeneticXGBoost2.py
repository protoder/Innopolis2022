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



class TXGBGenetic(TMLGenetic):  # TDistributedGenetic):
    def __init__(self, X, Y, XTest, YTest, Seed=None, Debug = False, StopFlag=2, UseMetric="mlogloss", Verbose = 1,
                 TheBestListSize=50, StartPopulationSize=50, PopulationSize=100):
        super().__init__(HromosomLen=12 + 2, X=X, Y=Y, XTest=XTest, Seed=Seed, Debug = Debug, FixedGroupsLeft=0,
                            StopFlag=StopFlag, TheBestListSize=TheBestListSize, StartPopulationSize=StartPopulationSize,
                         PopulationSize=PopulationSize)
        self.InverseMetric = False  # т.е. опимизация на возрастание
        self.Metric = 0
        self.UseMetric = UseMetric

        self.FitParams = {'X':None, 'y':None}
        self.StoredPath = f'{CrPath}copy/XGB4copy.dat'
        self.Verbose = Verbose
        self.FixedEstimators = True
        self.ResFileName = 'XGB4'
        self.TryLoadOnStart = True
        self.MetricLimitForPublic = 0.937

    def GetModel(self, Hr):
        RA = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100]
        Methods = ['exact', 'approx', 'hist']
        CrMethod = Methods[int(Hr[1] * 3)]

        if self.GPU:  # and CrMethod == 'hist':
            CrMethod = 'gpu_hist'

        return XGBClassifier(learning_rate=0.01,
                             n_estimators=40 if self.FixedEstimators else int(Hr[9] * 10000),
                             max_depth=int(Hr[4] * 7) + 3,
                             min_child_weight=int(Hr[5] * 5) + 1,
                             max_bin=int(Hr[0] * 100) + 5,
                             gamma=Hr[6] / 2,
                             subsample=Hr[7] * 0.4 + 0.6,
                             colsample_bytree=Hr[8] * 0.4 + 0.6,
                             reg_alpha=RA[int(Hr[1] * 14)],  # 0.005,
                             objective='binary:logistic',
                             nthread=6,
                             scale_pos_weight=1,
                             seed=int(Hr[3] * 65535),
                             tree_method=CrMethod,
                             random_state=int(Hr[3] * 65535),
                             verbose=self.Verbose
                             , silent=True)

    def GetMetric(self, History, model):
        return None

    def BeforeFit(self, X, Y, ValidX, ValidY, Hr, FitParams, Verbose):
        eval_set = [(ValidX, ValidY)]

        if not self.FixedEstimators:
            es = EarlyStopping(
                rounds=15,
                save_best=True,
                maximize=False,
                data_name="validation_0",
                metric_name=self.UseMetric)
            FitParams['callbacks'] = [es]

        FitParams['eval_metric'] = self.UseMetric
        FitParams['eval_set'] = eval_set
        FitParams['verbose'] = Verbose>0


        return FitParams



random.seed(25)

X, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True)
X0, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
ResX0, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True)
ResX1, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)
# Y = Y[:, 1]
# ResY = ResY[:, 1]


if True:
    Gn = TXGBGenetic(X, Y[:, 1:2], ResX0, ResY[:, 1:2], StopFlag=0, Verbose = 0)
    Gn.PDeath = 0.75
    Gn.PMutation = 0.75  # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
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