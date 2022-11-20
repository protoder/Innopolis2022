from sktime.classification.kernel_based import RocketClassifier
from Libs import * #ReadCsv, WriteCsv, Graphic, Filter
from Experiments import *
from NN import *

CrPath = "E:/Uinnopolis/"

X, Y = ReadCsv(CrPath, DelZeros = True, SortCtg = False, Train = True, RetPrc = True)
XTest, YTest = ReadCsv(CrPath, DelZeros = True, SortCtg = True, Train = False, Au = True)

X_train, X_test, y_train, y_test = train_test_split(X, Y[:, 1], test_size=0.1, random_state=42)

if False:
    rocket = RocketClassifier(num_kernels = 1000)
    rocket.fit(X_train, y_train)
    y_pred = rocket.predict(X_test)
    print(recall_score(y_test, y_pred, average="macro", zero_division=0))
    print(0, classification_report(y_pred, y_test))
    WriteCsv(fr'e:/musor/Rocet.csv', y_test[:, 1:2], y_pred)
    WriteCsv(fr'e:/musor/Rocet.csv', y_test[:, 1:2], y_pred)

from sktime.classification.hybrid import HIVECOTEV2
hc2 = HIVECOTEV2(time_limit_in_minutes=1)
hc2.fit(X_train, y_train)
y_pred = rocket.predict(X_test)
print(recall_score(y_test, y_pred, average="macro", zero_division=0))
print(0, classification_report(y_pred, y_test))
WriteCsv(fr'e:/musor/HV.csv', y_test[:, 1:2], y_pred)