# -*- coding: utf-8 -*-
"""Поля_baseline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VxIsXZyI6Gqgd2ZgPZChIw32ywZESY5B

## Загрузим нужные библиотеки
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import Libs

# %matplotlib inline

#from google.colab import drive

    # Подключаем Google drive
#drive.mount('/content/drive')

df = pd.read_csv("E:/Uinnopolis/Data/train.csv")
TestDf = pd.read_csv("E:/Uinnopolis/Data/test.csv")

"""## Рассмотрим датасет по ближе"""

df.head(3)

df.info()

"""Рассмотрим единственный не числовой столбец"""

n = df.to_numpy()
print(n[2:38])

import matplotlib.pyplot as plt
plt.plot(n[2, 2:38])
plt.show()

df.select_dtypes(include=['object'])

"""Это столбец **.geo**

Тепепрь рассмотрим главный столбец с C/Х культурой
"""

df["crop"].hist(bins = 7)

"""Заметен слабый разброс популярности категориий"""

df["crop"].value_counts()

"""Посмотрим на разброс площади полей"""

sns.countplot(x = "area" , data  = df)

"""Постараемся найти закономерность"""

sns.jointplot(x = "crop", y = 'area', data = df, kind = 'reg')

"""Кажется нет четкой зависимости между категорией культуры и площадью территроии

Последняя попытка найти явную зависимость в данных
"""

plt.rcParams['figure.figsize']=(15,15)

corr = df.loc[:, "nd_mean_2021-08-27":"crop"].corr()
g = sns.heatmap(corr, square = True, annot=True)

"""## Выделим выборки"""

X = df.drop(["id",".geo", "crop"], axis = 1)
y = df[["crop"]]

X.head()

"""## Обучение модели"""
#from sktime.datasets import load_arrow_head
#from sktime.classification.compose import TimeSeriesForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#clf = TimeSeriesForestClassifier()
clf = RandomForestClassifier(random_state=0, n_estimators = 3)

clf.fit(X_train, y_train)

"""## Оценка точности"""

from sklearn.metrics import recall_score, precision_score

pred = clf.predict(X_test)
print(clf,'\n',recall_score(y_test, pred, average="macro", zero_division=0))
a = 0
