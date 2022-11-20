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

from Libs import * #ReadCsv, WriteCsv, Graphic, Filter
from Experiments import *
from NN import *
from BaseGenetic import *

'''
    Вместе с 
'''

Debug = False

class TForestGenetic(TBaseGenetic):#TDistributedGenetic):
    #def __init__(self, X, Y, Paths, HrCount, Seed = None, StopFlag= 2):
    def __init__(self, X, Y, Seed=None, StopFlag=2):
        TBaseGenetic.__init__(self, HromosomLen = 7 + 2, FixedGroupsLeft=0, StopFlag= StopFlag, TheBestListSize = 200, StartPopulationSize = 25, PopulationSize = 50)
        np.random.seed(Seed)
        self.X, self.Y = X, Y
        self.InverseMetric = False # т.е. опимизация на возрастание
        self.Metric = 0

    def TestHromosom(self, Hr, Id):
        if not Debug:
            Predict, CrValue = ForestHromosom(Hr, self.X, self.Y, None )
        else:
            CrValue = 1 - random.random() / 100
        return CrValue

    def GenerateHromosom(self, GetNewID = True):
        return np.random.random(size = self.HromosomLen)

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
X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
Gn = TForestGenetic(X, Y[:, 1], StopFlag = 0)
Gn.PDeath = 0.75
Gn.PMutation = 0.5 # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
Gn.PMultiMutation = 0.3
Gn.PCrossingover = 0.5
Gn.Start()


