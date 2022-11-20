import numpy as np
import random
from tensorflow.keras.models import Model, load_model  # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, SpatialDropout2D, MaxPooling2D, \
    AveragePooling2D, Conv2D, BatchNormalization  # Импортируем стандартные слои keras
from tensorflow.keras import backend as K  # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Adam  # Импортируем оптимизатор Adam
from tensorflow.keras import \
    utils  # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
from keras import regularizers, initializers

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
    Описание сети
        0 - Кол-во слоев (0..8), оптимизатор 
        Далее слои
        
        Слой:
            Количество нейронов (до 10000)
            Kernel регуляризация
            Activate регуляризация
            Инициализация весов - seed value
            Dropout, Batch 
            Активация
'''
activation_list = ['relu', 'elu', 'selu', 'tanh', 'sigmoid']
optimizer_list = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']

DensesStart = 1
DenseLen = 5
# размер = 41

def Regularisation(Gene, Name):
    if Gene <= 0.84:
        reg = None
        Reg = ''
    elif Gene[1] <= 0.92:
        reg = regularizers.L1(4*(Gene - 0.84))# максимум 0.32
        Reg = f', {Name}_reg = L1({4*(Gene - 0.84)})'
    else:
        reg = regularizers.L1(4 * (Gene - 0.92))  # максимум 0.32
        Reg = f', {Name}_reg = L1({4 * (Gene - 0.92)})'

    return reg, Reg

'''def GetWeithsIn(Gene, Min, Max):
    initializers.RandomNormal(mean=(Min + Max)/2, stddev=0.5, seed=None)
    initializers.RandomUniform(minval=0., maxval=1.)
'''

Debug = False

def DenseNetFromHromosom(Hr, XShape, YShape = 7, metrix =['accuracy']):
    Inp = Input(shape=(XShape[-1]))
    X = Inp
    Levels = Hr[0] & 7
    Pos = 1
    Txt = ''
    for Level in range(Levels):
        KernelReg, KernelRegStr = Regularisation(Hr[Pos + 1], 'Kernel')
        ActiveReg, ActiveRegStr = Regularisation(Hr[Pos + 2], 'Active')

        Norm = Hr[Pos + 4]
        ActFlag = Hr[Pos + 5]
        InplaceAct = True
        if ActFlag > 0.5:
            ActFlag-= 0.5
            InplaceAct = False

        ActivationName = activation_list[int(ActFlag * 10)]

        Str = ''

        if InplaceAct:
            InplaceActivationName = ActivationName
        else:
            InplaceActivationName = None

            X = Dense(Hr[Pos], activity_regularizer = ActiveReg, kernel_regularizer = KernelReg, activation = InplaceActivationName,
                  kernel_initializer=initializers.GlorotUniform(seed = int(Hr[Pos + 3] * 65535)))(X)

            Str = f'{Str}, Dense({Hr[Pos]}, {KernelRegStr}, {ActiveRegStr}, ' \
                  f'{InplaceActivationName if InplaceActivationName is not None else ""}'
        if Norm > 0.5:

            if Norm < 0.75:
                X = BatchNormalization()(X)
                Str = f'{Str}, BN'
            else:
                X = DropOut(Norm - 0.75)(X)
                Str = f'{Str}, {DropOut(Norm - 0.75)}'

        if not InplaceAct:
            X = Activation(ActivationName)(X)
            Str = f'{Str}, {ActivationName}'

    X = Dense(YShape, activation="softmax")(X)

    model = Model(inputs=Inp, outputs=XDENSE)
    model.compil(loss='categorical_crossentropy', optimizer=optimizer_list[int(Hr[0] * 7 / 8)], metrics=metrics)

    model.summary()

    return


class TNNGenetic(TBaseGenetic):#TDistributedGenetic):
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

random.seed(25)

X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
X0, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True, PostProc = True)
ResX0, ResY = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = False, RetPrc = True)
ResX1, ResY = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = False, RetPrc = True, PostProc = True)

class THromosomCallback(Callback):
    def __init__(self):
        self.State = [0, 0]  # лучшее значение, его позиция
        self.BestW = None

    def on_epoch_end(self, epoch, logs=None):
        CR = logs['accuracy']
        if (CR == None) or (CR < 0.01) :
            print('Abort learning, accuracy=', logs['accuracy'])
            #print('Abort learning, acc=', logs['accuracy'], file=f)

            self.model.stop_training = True
            return


        if CR > self.State[0]:
            self.State[0] = CR
            self.State[1] = epoch

            #self.BestW = self.model.get_weights()



Epochs = 10
Batch_size = 64
verbose = 1
ModelsPath = 'NNModels/'
for i in range(10000):
    Hr = np.random.random(41 + 5)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f'{ModelsPath}Dense{i}.h5',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    Start = int(Hr[0] * (len(X) - 100))

    X1, Y1 = TimeAugmentation(np.delete(X, Start + np.arange(100), axis=0),
                              np.delete(Y, Start + np.arange(100), axis=0),
                              K=1 + int(Hr[1] * 30), random=int(Hr[4]*65535),
                              LevelK=0.01 + Hr[2] / 3.3, UseMain=Hr[3]>0.9)

    ValidX, ValidY = TimeAugmentation(X[Start:Start + 100],
                                    Y[Start:Start + 100],
                                    K=20, random=24, LevelK=0.1, UseMain=False)

    Model = DenseNetFromHromosom(Hr[5:], X1.shape)

    InfoCallBack = THromosomCallback
    History = Model.fit(X, Y, callbacks = [model_checkpoint_callback, InfoCallBack],
                        epochs= Epochs, batch_size= Batch_size, verbose= verbose,
                        validation_data=(ValidX, ValidY))

    print('Лучший результат ', InfoCallBack.State[0], 'эпоха', InfoCallBack.State[1])

    plt.figure(figsize=(14, 7))

    if 'val_accuracy' in History.history:
        plt.plot(History.history['val_accuracy'],
                 label='Доля верных ответов на тестовом наборе')
    plt.plot(History.history['accuracy'],
             label='Доля верных ответов на тренировочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()

'''Folder = fr'{CrPath}/Models/'
X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
Gn = TForestGenetic(X, Y[:, 1], StopFlag = 0)
Gn.PDeath = 0.5
Gn.PMutation = 0.5 # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
Gn.PMultiMutation = 0.3
Gn.PCrossingover = 0.5
Gn.Start()'''


