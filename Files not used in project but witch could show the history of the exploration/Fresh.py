# для построения моделей воспользуемся sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from Libs import * #ReadCsv, WriteCsv, Graphic, Filter
from Experiments import *
from NN import *

CrPath = "E:/Uinnopolis/"

def LoadFeatures(FileName, DataWithFeatures):
  Fch = np.load(FileName, allow_pickle=True)
  Fch = eval(str(Fch))
  #print(1, len(Fch), Fch)

  DelFch = [i for i in np.arange(len(Fch)) if i not in Fch]
  #print(2, len(DelFch), DelFch)

  Res = np.delete(DataWithFeatures, DelFch, -1)
  #print(3, len(DataWithFeatures), DataWithFeatures)
  #print(4, len(Res), Res)
  #print('Удалено фич ', len(DataWithFeatures) - len(Res))


  return Res

def TestForest(X, Y, N = 175):
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1)

  random.seed(1)
  Frst = RandomForestClassifier(random_state=0, n_estimators=N)
  R = Frst.fit(X_train, y_train[:,1])
  pred = Frst.predict(X_test)
  print(recall_score(y_test[:,1], pred, average="macro", zero_division=0))
  print(0, classification_report(pred,y_test[:,1]))


X1 = np.load(CrPath + 'DS.npy')
XTst = np.load(CrPath + 'DSTst.npy')

Fch = np.load(CrPath + 'XfeaturesS.npy', allow_pickle=True)
Fch = eval(str(Fch))
NF = np.load(CrPath + 'Xfeatures.npy')

XTemp, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, Au = True)

XF0 = np.load(CrPath + 'DS.npy') # неочищенные фичи
XS = np.concatenate([XF0,XTemp], -1)

X = np.load(CrPath + 'Xfeatures.npy')
XFull = np.load(CrPath + 'XfeaturesFull.npy')

print('Тест временного ряда')
TestForest(XTemp, Y, N = 175)

print('Тест по неочищенным фичам')
TestForest(XF0, Y, N = 175)

print('Тест временного ряда с неочищенными фичами')
TestForest(XS, Y, N = 175)

print('Тест по очищенным фичам')
TestForest(X, Y, N = 175)

print('Тест по очищенным фичам с временными рядом')
TestForest(XFull, Y, N = 175)



X0, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, Au = True)

X = np.concatenate([X1,X0], -1)

DelFch = [i for i in np.arange(len(Fch)) if i not in Fch]


X0 = np.delete(X, DelFch, -1)

X_train, X_test, y_train, y_test = train_test_split(X0, Y, test_size=.1)

random.seed(1)
Frst = RandomForestClassifier(random_state=0, n_estimators=175)
R = Frst.fit(X_train, y_train[:,1])
pred = Frst.predict(X_test)
print(recall_score(y_test[:,1], pred, average="macro", zero_division=0))
print(0, classification_report(pred,y_test[:,1]))


relevant_features = set()

#from tsfresh import select_features

for label in range(7):
    # select_features работает с бинарной классификацией, поэтому переводим задачу
    # в бинарную для каждого класса и повторяем по всем классам
    y_train_binary = Y == label
    DF = pd.DataFrame(X)
    #X_train_filtered = select_features(DF, y_train_binary)
    relevant_features = relevant_features.union(set(X_train_filtered.columns))

len(relevant_features)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1)



random.seed(1)
Frst = RandomForestClassifier(random_state=0, n_estimators=175)
R = Frst.fit(X_train, y_train[:,1])
pred = Frst.predict(X_test)
print(recall_score(y_test[:,1], pred, average="macro", zero_division=0))
print(0, classification_report(pred,y_test[:,1]))

#DecisionTreeClassifier
'''
for N in range(40):
    random.seed(1)
    Frst = RandomForestClassifier(random_state=0, n_estimators=25 + N*25)
    R = Frst.fit(X_train, y_train[:,1])
    pred = Frst.predict(X_test)
    print(25 + N*25, recall_score(y_test[:,1], pred, average="macro", zero_division=0))

25 0.824359942426838
50 0.8482564424322089
75 0.8593678282079906
100 0.8571007004249341
125 0.862568013908552
150 0.8698671663689946
175 0.8790216063990107
200 0.8692752052762571
225 0.8775482285863472
250 0.8757883942178172
275 0.8777819826770437
300 0.8757883942178172
325 0.87802053707496
350 0.8740697259812918

'''
a =0
