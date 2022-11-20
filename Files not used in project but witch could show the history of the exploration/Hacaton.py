import re
import pickle
import pandas as pd
import numpy as np
import os, glob

from flask import Flask

app = Flask(__name__)


@app.route('/')

'''
    Разбиваем строку на слова
'''
def ExtractWords(S):
    res = re.findall(r'\w+', S)
    return res

'''
    Создаем словарь слов
'''
def ExtractFromList(L, FileName):

    Dict = {}

    for Razdels in L:
        for Works in Razdels:
            for Items in Works:
                Words = ExtractWords(Items[2]) # вытаскиваем слова из названия

                for CrWord in Words:
                    if CrWord in Ditc:
                        WordItems = Ditc[CrWord]
                    else:
                        WordItems = []

                        Ditc[CrWord] = WordItems

                    CrLine = [Col for Col in Items]

                    WordItems.append(CrLine)


    with open(FileName, 'wb') as FileVar:
        pickle.dump(Obj, FileVar)


'''
   Грузим словари справочников по списку
  
'''
DicList = {Name: Path}

def LoadDicts(DicList):

    ObjList = {}

    for Name in DicList:

        Path = DicList[Name]

        with open(Path, 'rb') as FileVar:
            Obj = pickle.load(FileVar)

            ObjList[Name] = Obj

    return ObjList

# Собственно анализ сметы

def Analize(FileName, DocList):

    L = Process(FileName)

    for Razdels in L:
        for Works in Razdels:
            for Items in Works:
                Words = ExtractWords(Items[2])

                Res = set()

                for CrWord in Words:

                    CrSet = set()

                    for Dic in DocList:

                        if CrWord in Dic:
                            CrRes = Dic[CrWord]

                            CrSet+= CrRes

                    if len(Res) == 0:
                        Res += CrSet
                    else:
                        Res *= CrSet

    return Res # вернет список строк

if __name__ == '__main__':
    app.run()