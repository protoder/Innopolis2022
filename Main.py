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
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

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



def Predict(InputCsv):

    # Читаем входной файл. Обрабатываем нули (DelZeros=True). Добавляем производные (PostProc=True)
    X_Test, y_test = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)

    y_test = y_test[:, 1:2] # Не поднялась рука сразу убрать из Y всю дополнительную информацию типа координат ( я так
                            # и не придумал, как ею пользоваться ), в итоге получился двумерный массив.
                            # Так что теперь выбираем нужное нам измерение - список позиций, которые пойдут в поле id
                            # итогового файла.

    # Получаем фичи, вместе с исходными жанными и производными почти 100 штук.
    Ftr_Test = GetFeatures(X_Test)
    XTest = np.concatenate((Ftr_Test, X_Test), -1)

    fInd = np.load(f'{CrPath}Input/FeatureIndex3.npy') # Загрузили маску наиболее информативных фич. Маска была подготовлена
                                 # раньше на основании анализа тестовых данных, см. ExtractBestFeatures.py
    XTest = XTest[:, fInd]
    # Все. Входные данные готовы.
    # Теперь запускаем ансамбль лучших классификаторов. Файл с описанием ансамбля сформирован ранее. См. SelectBestAnsamble.py
    # Модели в ансамбле предварительно подбирались генетическим алгоритмом ( подробнее об этом в описании или комментариях
    # к SelectBestAnsamble.py ). Поэтому они представлены "хромосомами" - массивом чисел, однозначно описывающим модель.
    # Как понимать хромосомы, знает процедура, которая будет передана в TryAnsamble (см. комментарии к UAnsambler.py ).
    # N - количество моделей. Последний элемент массива - рейтинги каждой модели
    # В нашем случае ProcessXGboost - обработчик классификатора градиентным бустингом

    ModelHrs = np.load(f'{CrPath}Input/Ansamble.npy')
    Res = TryAnsamble(XTest, ModelHrs, ProcessXGboost)

    # Формируем выходной файл.
    ResFile = f'{CrPath}Predict.csv'
    WriteCsv(ResFile , y_test, Res)

    print('Сформирован', ResFile)

if __name__ == '__main__':
    Predict(f'{CrPath}Data/test.csv')