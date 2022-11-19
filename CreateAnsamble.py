'''
    Главный метод - predict(InputCsv)
        Производит классификацию входного файла

    Требования к среде.
        1. Запуск необходимо запускать в среде с активной CUDA и графическим процессором. Локальная система у меня не столь
           богата, поэтому файл рассчитан на запуск в Colab. При отсутствии GPU он переходит в отладочный режим
        2. В зоне прямой видимости должен быть каталог Input
        3. Переменная CrPath должна быть установлена на каталог программы (понимаю, что лажа. Но нет времени воевать с
           colab )
        4. Перед началом объявлений import переменных идет закомментированным набор команд, необходимый для быстрого
           развертывания в среде Colab
'''

'''
# Команды инсталяции в Google Colaboratory

%pip install tsfresh

# По умолчанию в Colab ( по крайней мере бесплатном) стоит очень древняя версия библиотеки. Придется ее заменить
# При запуске удаления Colab запросит подтверждения. Не забудьте ответить
%pip uninstall xgboost
%pip install xgboost==1.6.2
import xgboost as xgb
xgb.__version__

'''

import numpy as np
import random
from numpy import loadtxt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import os
import os.path
from tqdm.auto import tqdm
import glob
import warnings
from tensorflow.test import is_gpu_available

GPU=is_gpu_available()

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

if not GPU:
    def GetFeatures(X, FileName=None):
        Res = np.load(f'{CrPath}Input/Ftr_test.npy')
        return Res
else:
    from xgboost import XGBClassifier
    from xgboost.callback import EarlyStopping
    import xgboost as xgb


    # Метод получения "фич" для набора данных.
    def GetFeatures(X, FileName=None):
        CrComprehensive = settings.ComprehensiveFCParameters()

        data_long = pd.DataFrame({0: X.flatten(), 1: np.arange(X.shape[0]).repeat(X.shape[1])})

        X = extract_features(data_long, column_id=1, impute_function=impute, default_fc_parameters=CrComprehensive)
        if FileName is not None:
            np.save(FileName, X)
        return X

import os, glob

from Libs import *
from UAnsambles import *


Alist = ['E:/Uinnopolis/TestModels/XGB_5_28013_0.9830.npy',
         'E:/Uinnopolis/TestModels/XGB_5_27461_0.9835.npy']

def CreateAnsambleFromList(List):
    Res = np.zeros((len(List), 13))
    for i, A in enumerate(List):
        try:
            Raiting = float(A[-10:-4])
        except:
            Raiting = 1


        Hr = np.append(np.load(A), Raiting)
        Res[i, :] = Hr

    np.save(f'{CrPath}Input/Ansamble.npy', Res)


def CreateAnsambleFromFolder(Mask):
    list = glob.glob(Mask)
    CreateAnsambleFromList(list)

'''
Как работает. 
1) Читаем все хромосомы. Получаем все predict в один массив
Далее начиная с последнего убираем. Если скор на Valid растет, так и отбрасываем его. Иначе оставляем.
'''
def OptimizeAnsamble(List, X, Y, XValid, YValid, Proc, Params = {}):
    _x, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
    y_valid = Y[:100]
    y_train = Y[100:]

    GPU = is_gpu_available()

    ModelsReiting = []

    K = 20
    y_train = np.tile(y_train, [K, 1])

    XValid = np.load(f'{CrPath}Input/Valid_100.npy')
    X = np.load(f'{CrPath}Input/Train_100.npy')

    y_valid = np.tile(y_valid, [K, 1])

    Res = np.zeros( (len(List, 7)) )
    HRList = np.zeros( (len(List, 13)) )
    R = np.zeros( (len(List)) )

    for i, Hr in enumerate(List):
        R[i] = Hr[-1:]
        Hr = Hr[:-1]

    Res = [0]*len(List)

    Accuracies = np.zero(len(List))

    for i, Hr in enumerate(List):
        R = Hr[-1:]
        Hr = Hr[:-1]
        test_pred, accuracy, y_pred, y_pred_cat = Proc(Hr, X, Y, ValidX=XValid, ValidY=y_valid, TestX=None, **Params)
        Accuracies[i] = accuracy
        Res[i] = (y_pred_cat - 0.5) * R

    Res = np.concatenate(Res)
    y_pred = Res.sum(0).argmax(-1)

    accuracy = accuracy_score(y_valid[:, 1:2], y_pred)
    print('Accuracy 0: %.8f%%' % (accuracy * 100.0))

    for Pos in range(len(Res), 0, -1):
        NewRes = np.delete(Res, [Pos])
        y_pred = NewRes.sum(0).argmax(-1)
        NewAccuracy = accuracy_score(y_valid[:, 1:2], y_pred)

        if NewAccuracy > Accuracy:
            Accuracy = NewAccuracy
            Res = NewRes



CreateAnsambleFromFolder('E:/Uinnopolis/TestModels/XGB_5_*.npy')


