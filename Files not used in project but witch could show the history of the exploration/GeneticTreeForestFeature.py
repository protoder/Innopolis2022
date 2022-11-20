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


def XGSearchFeatures(SearchFtrX, SearchFtrY, X, Y, ValidX, ValidY, XTest, From = 0, InputMask = None, GPU = 0, UseMetric="mlogloss"):

    def StartModel(Ft, PrintI):
        model = RandomForestClassifier(random_state=232, n_estimators=127)

        VX = ValidX[:, Ft]
        eval_set = [(VX, ValidY)]

        model.fit(X[:, Ft], Y)

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

    ValidX, ValidY = TimeAugmentation(ValidX, ValidY,
                                      K=20, random=24, LevelK=0.05, UseMain=True)

    if not os.path.isfile('forest_feature_importances_.npy'):
        model = RandomForestClassifier(random_state=232, n_estimators=127)

        model.fit(X, Y)

        f = model.feature_importances_

        np.save('forest_feature_importances_.npy', f)
    else:
        f = np.load('forest_feature_importances_.npy')

    fInd = np.argsort(f)[::-1]
    CrIndex = fInd.copy()
    LastAccuracy = 0

    if not os.path.isfile('ForestResult.npy'):
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
                np.save(f'ForestResult{n}.npy', Ft)
            #print(i, 'Accuracy: %.8f%%' % (accuracy * 100.0))



        Ft = np.array(Result)
        np.save('ForestResult.npy', Ft)

    else:
        Ft = np.load('ForestResult.npy')

    HallIndexes = np.arange(len(Ft))
    #StartModel(HallIndexes[-70:], 'Тест на исходных')

    Raising = [i == 0 or Ft[i]>Ft[i-1] for i in range(len(Ft))]
    Falling = 1 - np.array(Raising)

    #StartModel(HallIndexes[-139:], 'Тест на исходных с производными')
    Raising = [i == 0 or Ft[i] > Ft[i - 1] for i in range(len(Ft))]
    Falling = 1 - np.array(Raising)

    CrReiting = Ft[-1]

    #acc = StartModel(fInd, 'Тест полного результата, new ')

    TheBest = np.max(Ft)
    BestIndex = np.argmax(Ft)

    np.where(Ft > 0)

    fInd = fInd[Ft > 0]

    np.save('ForestFeatureIndex0.npy', fInd[:BestIndex])

    StartModel(fInd[:BestIndex], 'Финальный тест')

    print('Лучший результат', TheBest)
    print('Полный результат', CrReiting)


    DelIndexes = np.nonzero(Falling)[0]

    CrIndexes = fInd.copy()
    for Ind in DelIndexes:
        CrIndexes[Ind] = 0
        acc = StartModel(CrIndexes, 'Удалена ' + str(Ind) + ', ' +  str(CrReiting) + ', new ')
        #acc1 = StartModel(CrIndexes[:BestIndex], 'Тест по Best, new ' )
        if Ind == 920:
            np.save('ForestResult2.npy', CrIndexes)
        if (acc >= CrReiting):# or (acc1 >= TheBest):
            print('Удаляем', Ind, CrReiting)#, ' / ', acc1, TheBest)

            for i in range(len(Ft) - Ind):
                CrReiting = StartModel(fInd[Ind+1:], Ind + i)
                print(i, 'Восстановлена последовательность от', Ind+1)

                np.save(f'ForestResult3_{Ind + 1}.npy', CrIndexes)
                Ft[Ind+1:] = CrReiting

                TheBest = np.max(Ft)
                BestIndex = np.argmax(Ft)


                Result[i] = accuracy

            Ft = np.array(Result)
            np.save('ForestResult.npy', Ft)

            CrReiting = acc


        else:
            CrIndexes[Ind] = Ind
            print('Оставляем', Ind, acc, CrReiting)#, ' / ', acc1, TheBest)



    np.save('ForestResult2.npy', CrIndexes) # получаем список подходящих индексов. Его тоже стоит проверить потом.


def PredictFeature(HrFile, FeaturesIndFile, GPU = 0, Win = 100, UseMetric="mlogloss", Verbose = 1):
   X0, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
   XF0 = np.load(CrPath + 'DS.npy')  # неочищенные фичи
   XTrain = np.concatenate([XF0, X0], -1)

   ResX1, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)
   XTst = np.load(CrPath + 'DSTst.npy')
   XTest = np.concatenate([XTst, ResX1], -1)

   Y = Y[:, 1]

   Hr = np.load(HrFile)
   Ft = np.load(FeaturesIndFile)

   test_pred, accuracy, y_pred, y_pred_cat = BoostHr(Hr, XTrain[:, Ft], Y, XTest[:, Ft], WinSize=Win, GPU=GPU, UseMetric=UseMetric, Verbose = Verbose)
   return test_pred, accuracy, y_pred, y_pred_cat

if __name__ == '__main__':

    ResX1, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True, PostProc=True)
    if False:
        HrFile = 'e:/res/XGB64_0.9970.npy'
        #GPU = is_gpu_available()
        test_pred, accuracy, y_pred, y_pred_cat = PredictFeature(HrFile, 'ForestFeatureIndex0.npy', GPU=GPU, Verbose = True)
        WriteCsv(HrFile.replace('.npy', f'_ftr_hist_{accuracy:0.4f}.csv' if GPU else f'_ftr_{accuracy:0.4f}.csv'),
                 ResY[:, 1:2],y_pred)
    else:
        random.seed(25)

        X, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True)
        X0, Y = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
        ResX0, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True)

        X_valid, y_valid = X0[:100], Y[:100]
        X_train, y_train = X0[100:], Y[100:]

        XF0 = np.load(CrPath + 'DS.npy')  # неочищенные фичи
        XTrain = np.concatenate([XF0, X0], -1)

        XTst = np.load(CrPath + 'DSTst.npy')
        XTest = np.concatenate([XF0, X0], -1)

        FtrTrain = np.load('Ftr_train_100.npy')
        XTrain = np.concatenate([FtrTrain, X_train], -1)

        FtrFull = np.load('Ftr_train_Full.npy')
        XFull = np.concatenate([FtrFull, X0], -1)

        FtrValid = np.load('Ftr_valid_100.npy')
        XValid = np.concatenate([FtrValid, X_valid], -1)

        FtrTest = np.load('Ftr_test.npy')
        XTest = np.concatenate([FtrTest, ResX0], -1)

        #Ft = np.load('FeatureIndex0.npy')
        #BestIndex = np.argmax(Ft)
        XGSearchFeatures(XFull, Y[:, 1], XTrain, y_train[:, 1], XValid, y_valid[:, 1:2], XTest = None, GPU=0, UseMetric="mlogloss")
