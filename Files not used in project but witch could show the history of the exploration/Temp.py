import numpy as np

fInd = np.load('E:/Uinnopolis/FeatureIndex0.npy')

FtrInp0 = np.load('E:/Uinnopolis/Input/Ftr_train_100_1.npy').astype('float32')
Inp0 = np.load('E:/Uinnopolis/Input/Train_100_1.npy').astype('float32')

Inp1 = np.load('E:/Uinnopolis/Input/Train_100 0.npy').astype('float32')
FtrInp1 = np.load('E:/Uinnopolis/Input/Ftr_train_100 0.npy').astype('float32')

Inp = np.concatenate((Inp0, Inp1))
FtrInp = np.concatenate((FtrInp0, FtrInp1))

Train = np.concatenate((FtrInp, Inp), -1)
np.save('E:/Uinnopolis/Input/Train_100_FullFields.npy', Train)

Train = Train[:, fInd]

np.save('E:/Uinnopolis/Input/Train_100.npy', Train)

FtrValid = np.load('E:/Uinnopolis/Input/Ftr_valid_100.npy').astype('float32')
Valid = np.load('E:/Uinnopolis/Input/Valid_100.npy').astype('float32')

Valid = np.concatenate((FtrValid, Valid), -1)
np.save('E:/Uinnopolis/Input/Valid_100_FullFields.npy', Valid)
Valid = Valid[:, fInd]

np.save('E:/Uinnopolis/Input/Valid_100.npy', Valid)
a = 0





