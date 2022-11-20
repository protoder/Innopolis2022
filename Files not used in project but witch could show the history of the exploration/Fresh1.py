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

X = np.load(CrPath + 'DS.npy')
XTst = np.load(CrPath + 'DSTst.npy')

_, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, Au = True)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1)
#DecisionTreeClassifier
for N in range(40):
    random.seed(1)
    Frst = RandomForestClassifier(random_state=0, n_estimators=25 + N*25)
    R = Frst.fit(X_train, y_train[:,1])
    pred = Frst.predict(X_test)
    print(25 + N*25, recall_score(y_test[:,1], pred, average="macro", zero_division=0))
a = 0
