'''
Модуль собирает единые входные данные из файлов фич и соответствующих им входов.
Идея в том, что подготовлены фичи по аугментированным данным. Чтобы быть уверенным, что 
они точно соответствуют друг другу, при выгрузке фич были выгружены и соответствующие данные.
При этом для ускорения работы Train данные обрабатывались на двух серверах. Соответственно они разбиты на две части
'''
import numpy as np
import os
import os.path
from tqdm.auto import tqdm
import glob

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
    sys.path.append('/content/drive/Hacaton')
else:
    Acer = not os.path.exists("E:/Uinnopolis/")
    CrPath = "C:/Uinnopolis/" if Acer else "E:/Uinnopolis/"

from Libs import *

# Файл со списком рабочих фич. Создается модулем AnaliseFeatures
fInd = np.load('E:/Uinnopolis/FeatureIndex0.npy')

# Файлы Train
FtrInp0 = np.load('E:/Uinnopolis/Input/Ftr_train_100_1.npy').astype('float32')
Inp0 = np.load('E:/Uinnopolis/Input/Train_100_1.npy').astype('float32')

Inp1 = np.load('E:/Uinnopolis/Input/Train_100 0.npy').astype('float32')
FtrInp1 = np.load('E:/Uinnopolis/Input/Ftr_train_100 0.npy').astype('float32')

# Объединяем Train данные
Inp = np.concatenate((Inp0, Inp1))
FtrInp = np.concatenate((FtrInp0, FtrInp1))

Train = np.concatenate((FtrInp, Inp), -1)

# Сохраняем Train с полным набором фич ( для экспериментов )
np.save('E:/Uinnopolis/Input/Train_100_FullFields.npy', Train)

Train = Train[:, fInd]

# Сохраняем Train с оптимальным набором фич
np.save('E:/Uinnopolis/Input/Train_100.npy', Train)


# Аналогичная обработка Validation данных.
FtrValid = np.load('E:/Uinnopolis/Input/Ftr_valid_100.npy').astype('float32')
Valid = np.load('E:/Uinnopolis/Input/valid_100_X.npy').astype('float32')

ValidY = np.load('E:/Uinnopolis/Input/valid_100_Y.npy').astype('float32')
ResX0, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=True, RetPrc=True, PostProc=True)
Y = np.tile(ResY[:100,1:2], (20, 1))

Valid = np.concatenate((FtrValid, Valid), -1)
np.save('E:/Uinnopolis/Input/Valid_100_FullFields.npy', Valid)
Valid = Valid[:, fInd]

np.save('E:/Uinnopolis/Input/Valid_100.npy', Valid)

# Обоработка Test здесь не производится. Они не аугментированы, а главное,
# ее необходимо рассчитывать на лету. Их обычным образом читаем из csv файла
'''ResX0, ResY = ReadCsv(CrPath, DelZeros=True, SortCtg=False, Train=False, RetPrc=True)
FtrTest = np.load('E:/Uinnopolis/Input/Ftr_test.npy')
Test = np.concatenate([FtrTest, ResX0], -1)
np.save('E:/Uinnopolis/Input/Test_FullFields.npy', Test)
Valid = Test[:, fInd]

np.save('E:/Uinnopolis/Input/Test.npy', Test)'''
a = 0