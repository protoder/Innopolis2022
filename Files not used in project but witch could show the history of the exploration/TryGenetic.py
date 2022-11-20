# -*- coding: utf-8 -*-
"""
Попытка интерполировать временной ряд степенным рядом через генетику
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np


df = pd.read_csv("E:/Uinnopolis/Data/train.csv")

Columns = df.columns.to_list()
Columns.sort()

df_labels = df.reindex(columns=Columns[1:3])
SortInd = np.argsort(df_labels.to_numpy()[:,1], axis = 0)
df = df.reindex(columns=Columns[4:]).to_numpy()[SortInd]



for x, d in enumerate(df):
    for y, dd in enumerate(d):
        if y == 0 or y == 69:
            continue

        if dd == 0:
            df[x, y] = (df[x, y - 1] + df[x, y + 1]) / 2
'''

'''
df_labels = df_labels.to_numpy()[SortInd]
"""## Рассмотрим датасет поближе"""



import BaseGenetic

def Polinom(x, A, x0 = 0):
    xx = x - x0
    NFact = 1
    x = 1
    Sum = 0
    for i, a in enumerate(A):
        if i > 0:
            NFact = NFact * i
            x*= xx


        if i % 2 == 0:
            Sum += a * x / (NFact)
        else:
            Sum -= a * x / (NFact)

    return Sum

def Polinom2(x, A, x0 = 0):
    xx = x - x0
    NFact = 1
    x = 1
    Sum = 0
    for i, a in enumerate(A):
        if i > 0:
            x*= xx

            #if i % 2:
            Sum += a * x / (i*i)
            #else:
            #    Sum -= a * x / (i * i)
    return Sum

class TPolinomGenetic(BaseGenetic.TBaseGenetic):
    def __init__(self, Data):
        BaseGenetic.TBaseGenetic.__init__(self, HromosomLen = 10, FixedGroupsLeft=0, StartPopulationSize = 25, StopFlag=2, PopulationSize = 50)

        Max = Data.max()
        Min = (Data[Data != 0]).min()
        self.Data = (Data - (Min + Max))/(2*(Max - Min))

    def TestHromosom(self, Hr, Id):
        Error = 0
        Step = 0
        for i, D in enumerate(self.Data):
            if D == 0:
                continue

            Error += abs(D - Polinom(i / 25, Hr))

            Step += 1

        return 1000 * Error / Step

    def GenerateHromosom(self, GetNewID = True):
        Res =  (np.random.random(size = self.HromosomLen) - 0.5) * 2
        self.InitNewHromosom(Res, GetNewID)
        return Res

Gn = TPolinomGenetic(df[0])
Gn.PDeath = 0.5
Gn.PMutation = 0.5 # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
Gn.PMultiMutation = 0.3
Gn.PCrossingover = 0.3
Gn.Start()

print('Поколений ', Gn.Generation)




'''
import matplotlib.pyplot as plt

for i in range(707):
    #plt.figure(figsize=(14, 7))
    plt.plot(df[i+693, 2:38], color='green')

plt.show()
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=0, n_estimators = 3)

clf.fit(X_train, y_train)

"""## Оценка точности"""

from sklearn.metrics import recall_score, precision_score

pred = clf.predict(X_test)
print(clf,'\n',recall_score(y_test, pred, average="macro", zero_division=0))