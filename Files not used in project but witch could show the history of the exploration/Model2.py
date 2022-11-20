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

random.seed(0)
# 0.944373328132043
#0.9410589149710825
#0.9451024938228455
#0.9308071051668431 sin
#0.9436062870836721
# 0.9448789795328352 - +1я производная
# 0.9392104626750599 - + вторая
#CrModel = TRandomForestClassifier(n_estimators = 175)
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
