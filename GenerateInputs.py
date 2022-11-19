import numpy as np
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

from Libs import *

def Test(X, Y, ValidX, ValidY):
        UseMetric = "mlogloss"

        CrMethod = 'exact'

        GPU = is_gpu_available()

        if GPU:  # and CrMethod == 'hist':
            CrMethod = 'gpu_hist'

        model = XGBClassifier(learning_rate=0.03,
                              n_estimators=50000,
                              max_depth=6,
                              min_child_weight=6,
                              max_bin=100,
                              gamma=0,
                              subsample=0.6,
                              colsample_bytree=0.6,
                              reg_alpha=0.005,
                              objective='binary:logistic',
                              nthread=6,
                              scale_pos_weight=1,
                              seed=int(65),
                              tree_method=CrMethod,
                              random_state=65,
                              verbose=1)

        # Call back обеспечивает своевременную остановку при начале переобучения
        es = EarlyStopping(
            rounds=100,
            save_best=True,
            maximize=False,
            data_name="validation_0",
            metric_name=UseMetric
        )

        VX = ValidX
        eval_set = [(VX, ValidY)]

        model.fit(X, Y, eval_metric=UseMetric, eval_set=eval_set,
                  verbose=1, callbacks=[es])

        # make predictions for test data
        y_pred = model.predict(VX)

        accuracy = accuracy_score(ValidY, y_pred)
        print('Accuracy: %.8f%%' % (accuracy * 100.0))

        return model

def GenInputs(Path = 'E:/Uinnopolis/'):
    # Файл оптимальный признаков
    fInd = np.load(Path + 'Input/FeatureIndex3.npy')

    # Файлы с признаками тренировочного набора создавались медленно - ведь тренировочный набор расширен аугментацией
    # Поэтому он разбит на 2 части. Собираем их. Грузим как файлы с фичами, так и сами аугментированные наборы данных
    FtrInp0 = np.load(Path + 'Input/Ftr_train_100_1.npy').astype('float32')
    Inp0 = np.load(Path + 'Input/Train_100_1.npy').astype('float32')

    Inp1 = np.load(Path + 'Input/Train_100 0.npy').astype('float32')
    FtrInp1 = np.load(Path + 'Input/Ftr_train_100 0.npy').astype('float32')

    Inp = np.concatenate((Inp0, Inp1))
    FtrInp = np.concatenate((FtrInp0, FtrInp1))

    Train = np.concatenate((FtrInp, Inp), -1)
    np.save(Path + 'Input/Train_100_FullFields.npy', Train) # для экспериментов сохраняем заодно полный набор фич

    Train = Train[:, fInd] # получили тренировочный набор

    np.save(Path + 'Input/Train_100.npy', Train)

    # Аналогичсно с валидационным
    FtrValid = np.load(Path + 'Input/Ftr_valid_100.npy').astype('float32')
    Valid = np.load(Path + 'Input/Valid_100_X.npy').astype('float32')

    Valid = np.concatenate((FtrValid, Valid), -1)
    np.save(Path + 'Input/Valid_100_FullFields.npy', Valid)
    Valid = Valid[:, fInd]

    np.save(Path + 'Input/Valid_100.npy', Valid)

    # Для верности тестируем данные. Должны получить точность такую же, как в финале выбора признаков (99% в нашем случае)

    # Y не сохраняли, и создаем их на лету с учетом аугментации
    # ReadCsv читает входные данные, и производит предобработку - интерполирует нули и добавляет производные
    _, YTest = ReadCsv(Path, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
    Y = np.tile(YTest[100:], [20, 1])
    ValidY = np.tile(YTest[:100], [20, 1])

    Y = Y[:,1:2]
    ValidY = ValidY[:,1:2]

    model = Test(Train, Y, Valid, ValidY)

    # ну и под конец тест на публичных данных
    X_Test, y_test = ReadCsv(Path, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)

    y_test = y_test[:, 1:2]  # Не поднялась рука сразу убрать из Y всю дополнительную информацию типа координат ( я так
    # и не придумал, как ею пользоваться ), в итоге получился двумерный массив.
    # Так что теперь выбираем нужное нам измерение - список позиций, которые пойдут в поле id
    # итогового файла.

    # Получаем фичи, вместе с исходными жанными и производными почти 100 штук.
    Ftr_Test = np.load(f'{Path}Input/Ftr_test.npy')
    XTest = np.concatenate((Ftr_Test, X_Test), -1)

    # раньше на основании анализа тестовых данных, см. ExtractBestFeatures.py
    XTest = XTest[:, fInd]

    predict=model.predict(XTest)
    WriteCsv(fr'e:/Test.csv', y_test, predict)

if __name__ == '__main__':
    GenInputs()







