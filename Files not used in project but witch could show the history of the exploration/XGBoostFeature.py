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
import os.path


def XGSearchFeatures(X, Y, ValidX, ValidY, XTest, From = 0, InputMask = None, GPU = 0, UseMetric="mlogloss"):

    def StartModel(Ft, PrintI):
        model = XGBClassifier(learning_rate=0.01,
                              n_estimators=500,
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
                              verbose=0)

        VX = ValidX[:, Ft]
        eval_set = [(VX, ValidY)]

        model.fit(X[:, Ft], Y, eval_metric=UseMetric, eval_set=eval_set,
                  verbose=0)

        # make predictions for test data
        y_pred = model.predict(VX)

        accuracy = accuracy_score(ValidY, y_pred)
        print(PrintI, 'Accuracy: %.8f%%' % (accuracy * 100.0))

        return accuracy

    '''if InputMask == None:
        InputMask = np.ones(len(X))'''

    RA = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100]
    Methods = ['exact', 'approx', 'hist']
    CrMethod = 'exact'

    if GPU:  # and CrMethod == 'hist':
        CrMethod = 'gpu_hist'



        '''
        X1, Y1 = TimeAugmentation(X,
                                  np.delete(Y, Start + np.arange(WinSize), axis=0),
                                  K=1 + int(Hr[1] * 20), random=int(Hr[4] * 65535),
                                  LevelK=0.01 + Hr[2] / 3.3, UseMain= True)#Hr[13] > 0.9)
    
        Y1 = Y1[:, 1:2]
    
        ValidX, ValidY = TimeAugmentation(X[Start:Start + WinSize],
                                          Y[Start:Start + WinSize],
                                          K=20, random=24, LevelK=0.1, UseMain=True)
    
        ValidY = ValidY[:, 1:2]
        '''

    ValidX, ValidY = TimeAugmentation(ValidX, ValidY,K=20, random=24, LevelK=0.05, UseMain=True)

    model = XGBClassifier(learning_rate=0.01,
                          n_estimators= 50000,
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
                          verbose=0)

    es = EarlyStopping(
        rounds=100,
        save_best=True,
        maximize=False,
        data_name="validation_0",
        metric_name=UseMetric
    )

    eval_set = [(ValidX, ValidY)]

    model.fit(X, Y, eval_metric=UseMetric, eval_set=eval_set,
              verbose=0, callbacks=[es])

    f = model.feature_importances_

    np.save('feature_importances_.npy', f)
    fInd = np.argsort(f)[::-1]
    CrIndex = fInd.copy()
    LastAccuracy = 0

    if not os.path.isfile('FeatureIndex0.npy'):
        Result = [0] * len(f)

        i = From

        for n in range(len(f) - From):

            accuracy = StartModel(fInd[:i+1], i)

            if LastAccuracy > accuracy:

                fInd = np.delete(fInd, [i])
                Result[n] = 0
                print('Пропускаем', 'accuracy остается ', LastAccuracy)
                #accuracy = StartModel(fInd[:i + 1], i)
            else:
                LastAccuracy = accuracy
                Result[n] = accuracy
                i+= 1

            if n % 100 == 99:
                Ft = np.array(Result)
                np.save(f'Result{n}.npy', Ft)
            #print(i, 'Accuracy: %.8f%%' % (accuracy * 100.0))



        Ft = np.array(Result)
        np.save('Result.npy', Ft)

    else:
        Ft = np.load('FeatureIndex0.npy')

    HallIndexes = np.arange(len(Ft))
    StartModel(HallIndexes[-70:], 'Тест на исходных')

    Raising = [i == 0 or Ft[i]>Ft[i-1] for i in range(len(Ft))]
    Falling = 1 - np.array(Raising)

    StartModel(HallIndexes[-140:], 'Тест на исходных с производными')
    Raising = [i == 0 or Ft[i] > Ft[i - 1] for i in range(len(Ft))]
    Falling = 1 - np.array(Raising)

    CrReiting = Ft[-1]

    acc = StartModel(fInd, 'Тест полного результата, new ')

    TheBest = np.max(Ft)
    BestIndex = np.argmax(Ft)

    print('Лучший результат', TheBest)
    print('Полный результат', CrReiting)


    DelIndexes = np.nonzero(Falling)[0]

    CrIndexes = fInd.copy()
    for Ind in DelIndexes:
        CrIndexes[Ind] = 0
        acc = StartModel(CrIndexes, 'Удалена ' + str(Ind) + ', ' +  str(CrReiting) + ', new ')
        #acc1 = StartModel(CrIndexes[:BestIndex], 'Тест по Best, new ' )
        if Ind == 920:
            np.save('Result2.npy', CrIndexes)
        if (acc >= CrReiting):# or (acc1 >= TheBest):
            print('Удаляем', Ind, CrReiting)#, ' / ', acc1, TheBest)

            for i in range(len(Ft) - Ind):
                CrReiting = StartModel(fInd[Ind+1:], Ind + i)
                print(i, 'Восстановлена последовательность от', Ind+1)

                np.save(f'Result3_{Ind + 1}.npy', CrIndexes)
                Ft[Ind+1:] = CrReiting

                TheBest = np.max(Ft)
                BestIndex = np.argmax(Ft)


                Result[i] = accuracy

            Ft = np.array(Result)
            np.save('Result.npy', Ft)

            CrReiting = acc


        else:
            CrIndexes[Ind] = Ind
            print('Оставляем', Ind, acc, CrReiting)#, ' / ', acc1, TheBest)



    np.save('Result2.npy', CrIndexes) # получаем список подходящих индексов. Его тоже стоит проверить потом.


def PredictFeature(HrFile, FeaturesIndFile, GPU = 0, Win = 100, UseMetric="mlogloss", Verbose = 1):
   X0, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
   XF0 = np.load(CrPath + 'DS.npy')  # неочищенные фичи
   XTrain = np.concatenate([XF0, X0], -1)

   ResX1, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)
   XTst = np.load(CrPath + 'DSTst.npy')
   XTest = np.concatenate([XF0, X0], -1)

   Y = Y[:, 1]

   Hr = np.load(HrFile)
   Ft = np.load(FeaturesIndFile)

   test_pred, accuracy, y_pred, y_pred_cat = BoostHr(Hr, XTrain, Y, XTest, WinSize=Win, GPU=GPU, UseMetric=UseMetric, Verbose = Verbose)
   return test_pred, accuracy, y_pred, y_pred_cat

if __name__ == '__main__':

    ResX1, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)
    if True:
        HrFile = 'e:/res/XGB64_0.9970.npy'
        GPU = is_gpu_available()
        test_pred, accuracy, y_pred, y_pred_cat = PredictFeature(HrFile, 'FeatureIndex0.npy', GPU=GPU, Verbose = True)
        WriteCsv(HrFile.replace('.npy', f'_ftr_hist_{accuracy:0.4f}.csv' if GPU else f'_ftr_{accuracy:0.4f}.csv'),
                 ResY[:, 1:2],y_pred)
    else:
        random.seed(25)

        X, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True)
        X0, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
        ResX0, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True)

        # Y = Y[:, 1]
        # ResY = ResY[:, 1]

        XF0 = np.load(CrPath + 'DS.npy')  # неочищенные фичи
        XTrain = np.concatenate([XF0, X0], -1)

        XTst = np.load(CrPath + 'DSTst.npy')
        XTest = np.concatenate([XF0, X0], -1)

        X_test, y_test = XTrain[-500:], Y[-500:]
        X_train, y_train = XTrain[:500], Y[:500]

        Ft = np.load('FeatureIndex0.npy')
        BestIndex = np.argmax(Ft)
        XGSearchFeatures(X_train, y_train[:, 1], X_test, y_test[:, 1], XTest=None, GPU=0, From=BestIndex,
                         UseMetric="mlogloss")
