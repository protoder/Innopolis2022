'''
    TryAnsambleFromPath(XTest, Path, Proc)
    Метод получает на вход массив входных данных, и путь к файлам хромосом. Формат * 0.9999.npy  Числа - оценка на
    валидационном (локальном) датасете
    Вторым параметром - процедура обработки хромосомы. Ее формат:

    Proc(HR, X, Y, ValidX, ValidY, TestX, Params)

    Процедура производит расчет TestX на моделе, описанной в HR. В процессе модель обучается на тренировочном наборе
    X, Y и валидационном ValidX, ValidY. Если TestX = None, только обучение
    Params - возможные дополнительные параметры
    Вернет Accuracy на валидационной модели, и если указан TestX, вернет еще и predict по Test

'''

# Обработка списка хромосом. Последний элемент хромосомы = ее рейтинг
import numpy as np
from tensorflow.keras.utils import to_categorical
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

def TryAnsamble(XTest, Hromosoms, Proc, Params = {}, NCat = 7):
    # Тестовые и валидационные данные (их Х часть) мы для скорости берем из уже готовых файлов, где они объединены с лучшими фичами , см.
    # ExtractBestFeatures.py
    # Из csv файла обучающего набора мы берем только Y

    _x, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)

    # Выделяем валидационный набор. Скромненько - но на деле он потом расширяется аугментацией, см. ExtractBestFeatures.py
    y_valid = Y[:100]
    y_train = Y[100:]
    y_train = y_train[:,1:2]# Не поднялась рука сразу убрать из Y всю дополнительную информацию типа координат ( я так
    y_valid =  y_valid[:,1:2]# и не придумал, как ею пользоваться ), в итоге получился двумерный массив.
                            # Так что теперь выбираем нужное нам измерение - список позиций, которые пойдут в поле id
                            # итогового файла.

    K = 20
    Y = np.tile(y_train, [K, 1]) # Входные обучающие и тестовые данные уже аугментированы. Соответственно,
    y_valid = np.tile(y_valid, [K, 1]) # аугментируем так же и сам Y

    # Грузим валидационные и тестовые данные. И те, и другие - 20х аугментация (см. Libs.TimeAugmentation() ) и ExtractBestFeatures.py
    XValid = np.load(f'{CrPath}Input/Valid_100.npy')
    X = np.load(f'{CrPath}Input/Train_100.npy')

    Res = [0]*(len(Hromosoms))

    for i, Hr in enumerate(Hromosoms):
        R = Hr[-1]

        Hr = Hr[:-1]
        test_pred, accuracy, y_pred, y_pred_cat = Proc(Hr, X, Y, ValidX = XValid, ValidY = y_valid, TestX = XTest, **Params)

        Res[i] = (y_pred_cat - 0.5) * R # смещаю, потому что иначе умножение на рейтинг влияет только на
                # единичные значения. То есть в случае нуля голос слабых позиций равенн голосу сильных. Непорядок

    Res = np.concatenate(Res)
    Res = Res.sum(0).argmax(-1)

    return Res

def TryAnsambleFromPath(XTest, Path, Proc):
    list = glob.glob(DirName + "xtest*.npy")

    HrList = []
    Raitings = np.array([])

    for File in list:
        try:
            Raiting = float(File[-10:-4])
        except:
            Raiting = 1

        HR = np.load(File)
        np.append(Raitings, Raiting)
        HR.reshape( (1, len(HR)))
        HrList.append(HR)


    return TryAnsamble(XTest, np.concatenate(HrList), Proc)

'''def TryAnsambleFromPath2(XTest, Path, Proc):
    list = glob.glob(DirName + "xtest*.npy")

    HrList = []
    Raitings = []

    for File in list:
        try:
            Raiting = float(File[-10:-4])
        except:
            Raiting = 1

        HR = np.load(File)
        Raitings.append(Raiting)
        HR = np.load(File)
        HR.reshape( (1, len(HR)))
        HrList.append(HR)


    _x, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)

    # Выделяем валидационный набор. Скромненько - но на деле он потом расширяется аугментацией, см. ExtractBestFeatures.py
    y_valid = Y[:100]
    y_train = Y[100:]
    y_train = y_train[:,1:2]  # Не поднялась рука сразу убрать из Y всю дополнительную информацию типа координат ( я так
    # и не придумал, как ею пользоваться ), в итоге получился двумерный массив.
    # Так что теперь выбираем нужное нам измерение - список позиций, которые пойдут в поле id
    # итогового файла.

    K = 20
    Y = np.tile(y_train, [K, 1])  # Входные обучающие и тестовые данные уже аугментированы. Соответственно,
    y_valid = np.tile(y_valid, [K, 1])  # аугментируем так же и сам Y

    # Грузим валидационные и тестовые данные. И те, и другие - 20х аугментация (см. Libs.TimeAugmentation() ) и ExtractBestFeatures.py
    XValid = np.load(f'{CrPath}Input/Valid_100.npy')
    X = np.load(f'{CrPath}Input/Train_100.npy')

    Res = np.zeros((len(Hromosoms), NCat))
    ArgRaitings = np.argsort(Raitings)[::-1]
    HrList = np.concatenate(HrList)[ArgRaitings]

    for x, Hr in enumerate(HrList):
        for y, Hr in enumerate(HrList - x):
            for i, Hr in enumerate(HrList - y):
        CrRes = Proc(Hr, X, Y, ValidX=XValid, ValidY=y_valid, TestX=None, **Params)

        Raiting[i] = CrRes
        Predicts[i, :] = (to_categorical(CrRes) - 0.5) * R  # смещаю, потому что иначе умножение на рейтинг влияет только на
        # единичные значения. То есть в случае нуля голос слабых позиций равенн голосу сильных. Непорядок

    Res = Res.sum(0).argmax(-1)

    return TryAnsamble(XTest, , Proc)
'''