# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:27:17 2015

@author: cruz
"""

class DataFiles(object):

    def __init__(self):
        self.x = 0
    #end of __init__ method

    def createFile(self, filename):
        myFile=open(filename,'w') # delect all data in file
        myFile.close()

    def addToFile(self, filename, var):
        myFile=open(filename,'a')
        for i in range(len(var)-1):
            myFile.write(str(int(var[i]))+',')

        myFile.write(str(int(var[len(var)-1]))+'\n')
        myFile.close()

    def addFloatToFile(self, filename, var):
        myFile=open(filename,'a')
        for i in range(len(var)-1):
            myFile.write(str(var[i])+',')

        myFile.write(str(var[len(var)-1])+'\n')
        myFile.close()

    def readFile(self, filename):
        myFile=open(filename,'r')
        line=myFile.readline()
        dataFile = []
        while line != "":
            data = line.split(',')
            dataInt = []
            for i in range(len(data)):
                dataInt.append(int(data[i]))

            dataFile.append(dataInt)
            line = myFile.readline()
        myFile.close()
        return dataFile

    def readFloatFile(self, filename):
        myFile=open(filename,'r')
        line=myFile.readline()
        dataFile = []
        while line != "":
            data = line.split(',')
            dataInt = []
            for i in range(len(data)):
                dataInt.append(float(data[i]))

            dataFile.append(dataInt)
            line = myFile.readline()
        myFile.close()
        return dataFile

#end of class DataFiles
