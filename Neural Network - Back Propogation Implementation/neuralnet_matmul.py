# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:01:42 2019

@author: cheth
"""



import sys
import numpy as np

def file_reader(input_file, label_cnt):
    in_file = open(input_file,"r")
    in_file_list = in_file.readlines()
    row_cnt = len(in_file_list)
    feature_cnt = len(in_file_list[0].split(','))
    X = np.zeros([row_cnt,feature_cnt], dtype = np.float128)
    ##Y = np.zeros([row_cnt], dtype = np.float128)
        
    for i in range(row_cnt):
        row = in_file_list[i].split(',')
        one_hot = np.zeros([1,label_cnt],dtype = np.float128)
        for j in range(feature_cnt):
            if j == 0 :
                label = int(row[j])
                one_hot[0][label] = 1.0
                
                if i == 0:
                    Y = one_hot
                else:
                    Y = np.concatenate((Y,one_hot), axis = 0)
                
                X[i][j] = 1.0
            else:
                X[i][j] = row[j]
                    
    return X,Y,feature_cnt


def initialize_params(Feature_Cnt, Hidden_Units, Label_Cnt, Init_Flag):
    
    if Init_Flag == 1:
        Alpha = np.zeros([Hidden_Units, Feature_Cnt], dtype = np.float128)
        Beta = np.zeros([Label_Cnt, Hidden_Units + 1], dtype = np.float128)
    else:
        Alpha = np.random.uniform(-0.1,0.1,(Hidden_Units, Feature_Cnt))
        Beta = np.random.uniform(-0.1,0.1,(Label_Cnt, Hidden_Units + 1))
    
    return Alpha, Beta



class NNForward :

    def __init__(self,data, true_label, alpha, beta):
        self.__X = data
        self.__Y = true_label
        self.__Alpha = alpha
        self.__Beta = beta
        self.__A = None
        self.__Z = None
        self.__B = None
        self.__YCap = None
        self.__J = None
        
        ##LINEAR FORWARD : Set A ###
        self.setA()
        
        ##SIGMOID FORWARD : Set Z ####
        self.setZ()
        
        ##LINEAR FORWARD : Set B ######
        self.setB()
        
        ##SOFTMAX FORWARD : Set Y #####
        self.setYCap()
        
        ##CROSS ENTROPY FORWARD : Set J #####
        self.setJ()
        
    def setX(self, data) :
        self.__X = data
             
    def getX(self) :
        return self.__X

    def setY(self, true_label) :
        self.__Y = true_label
             
    def getY(self) :
        return self.__Y

    def setAlpha(self, alpha) :
        self.__Alpha = alpha
             
    def getAlpha(self) :
        return self.__Alpha  

    def setBeta(self, beta) :
        self.__Beta = beta
             
    def getBeta(self) :
        return self.__Beta 

    def setA(self) :           
        self.__A = np.matmul(self.__Alpha,self.__X.transpose())
             
    def getA(self) :
        return self.__A

    def setZ(self) :
        self.__Z = np.insert(1 / (1 + np.exp(-self.__A)), 0, 1.0)
             
    def getZ(self) :
        return self.__Z
        
    def setB(self) :
        self.__B = np.matmul(self.__Beta,self.__Z.transpose())
        
             
    def getB(self) :
        return self.__B

    def setYCap(self) :
        self.__YCap = np.exp(self.__B) / sum(np.exp(self.__B))
             
    def getYCap(self) :
        return self.__YCap
        
    def setJ(self) :
        self.__J = -1 * np.matmul(self.__Y , np.log(self.__YCap))
             
    def getJ(self) :
        return self.__J


class NNBackward :

    def __init__(self,NN_Forward):
        self.__NN_Forward = NN_Forward
        self.__X = NN_Forward.getX()
        self.__Y = NN_Forward.getY()
        self.__Alpha = NN_Forward.getAlpha()
        self.__Beta = NN_Forward.getBeta()
        self.__gJ = None
        self.__gYCap = None
        self.__gBeta = None
        self.__gZ = None
        self.__gA = None
        self.__gAlpha = None
        self.__gX = None
        
        ## set gJ ############
        self.setgJ()
        
        ## CROSS ENTROPY BACKWARD : set gYCAP ###
        self.setgYCap()
        
        ## SOFTMAX BACKWARD : SET gB ############
        self.setgB()
        
        ## LINEAR BACKWARD : SET gBeta ##########
        self.setgBeta()
        
        
        ## LINEAR BACKWARD : set gZ #############
        self.setgZ()
        
        ## SIGMOID BACKWARD : set gA ############
        self.setgA()
        
        ## LINEAR BACKWARD : set gAlpha #########
        self.setgAlpha()
        
        ## LINEAR BACKWARD : set gX #############
        self.setgX()

    def setX(self, data) :
        self.__X = self.__NN_Forward.getX()
             
    def getX(self) :
        return self.__X

    def setY(self, true_label) :
        self.__Y = self.__NN_Forward.getY()
             
    def getY(self) :
        return self.__Y

    def setAlpha(self, alpha) :
        self.__Alpha = self.__NN_Forward.getAlpha()
             
    def getAlpha(self) :
        return self.__Alpha  

    def setBeta(self, beta) :
        self.__Beta = self.__NN_Forward.getBeta()
             
    def getBeta(self) :
        return self.__Beta 

    def setgJ(self) :
        self.__gJ = 1.0
             
    def getgJ(self) :
        return self.__gJ
    
    def setgYCap(self) :
        self.__gYCap = None
             
    def getgYCap(self) :
        return self.__gYCap

    def setgB(self) :
        self.__gB = self.__NN_Forward.getYCap() - self.__NN_Forward.getY()
             
    def getgB(self) :
        return self.__gB

    def setgBeta(self) :
        self.__gBeta = np.matmul(self.__gB.reshape((len(self.__gB), 1)), self.__NN_Forward.getZ().reshape((1,len(self.__NN_Forward.getZ()))))
             
    def getgBeta(self) :
        return self.__gBeta
    
    def setgZ(self) :
        self.__gZ = np.matmul(self.__gB.reshape((1, len(self.__gB))),self.__NN_Forward.getBeta()[:,1:] )
             
    def getgZ(self) :
        return self.__gZ
    
    def setgA(self) :
        
        Z_TERM = np.matmul(self.__NN_Forward.getZ()[1:].reshape((len(self.__NN_Forward.getZ()[1:]), 1)), (1 - self.__NN_Forward.getZ()[1:].reshape((1,len(self.__NN_Forward.getZ()[1:])))))
        self.__gA = (np.matmul(self.__gZ,Z_TERM )).transpose()
             
    def getgA(self) :
        return self.__gA
    
    def setgAlpha(self) :
        self.__gAlpha = np.matmul(self.__gA,self.__NN_Forward.getX().reshape((1,len(self.__NN_Forward.getX()))))
             
    def getgAlpha(self) :
        return self.__gAlpha
    
    def setgX(self) :
        self.__gX = (np.matmul(self.__gA.transpose(), self.__NN_Forward.getAlpha()[:,1:])).transpose()
             
    def getgX(self) :
        return self.__gX


def Avg_Cross_Entropy(X, Y, alpha, beta):
    cross_entropy = 0.0
    for i in range(len(X)):
        NNF = NNForward(X[i], Y[i], alpha, beta)
        cross_entropy = cross_entropy + NNF.getJ()
       
    avg_CE = cross_entropy/len(X)
    
    return avg_CE
   
    
def Predict(X, Y, alpha, beta, metrics_out, label_file, data_type):
    error = 0.0
    
    
    lbl_file = open(label_file,"w")
    
    for i in range(len(X)):
        NNF = NNForward(X[i], Y[i], alpha, beta)
        lbl_file.write(str(np.argmax(NNF.getYCap())) + "\n")
        if np.argmax(Y[i]) != np.argmax(NNF.getYCap()):
            error = error + 1
        
    avg_error = error/len(X)
    
    lbl_file.close()
    
    metrics_file = open(metrics_out,"a")
    metrics_file.write("error(" +data_type + "): " + str(avg_error) + "\n")
    metrics_file.close()
    


def SGD(trainX, trainY, testX, testY, Feature_Cnt, Label_Cnt, hidden_units, init_flag, num_epoch, metrics_out):
    ##SGD(Train_Data, Train_Label, Test_Data, Test_Label, Feature_Cnt,label_cnt,  hidden_units, init_flag, num_epoch)
    Alpha, Beta = initialize_params(Feature_Cnt, hidden_units, Label_Cnt, init_flag)
    metrics_file = open(metrics_out,"w")
    for e in range(num_epoch):
        for i in range(len(trainX)):
            NN_Forward = NNForward(trainX[i], trainY[i], Alpha, Beta)
    
            NN_Backward = NNBackward(NN_Forward)
            
            Alpha = Alpha - (learning_rate * NN_Backward.getgAlpha())
            Beta = Beta - (learning_rate * NN_Backward.getgBeta())
            
        train_avg_CE = Avg_Cross_Entropy(trainX, trainY, Alpha, Beta)
        test_avg_CE = Avg_Cross_Entropy(testX, testY, Alpha, Beta)
        
        
        metrics_file.write("epoch="+ str(e + 1) + " crossentropy(train): "+str(train_avg_CE) + "\n")
        metrics_file.write("epoch="+ str(e + 1) + " crossentropy(test): "+str(test_avg_CE) + "\n")
        
    metrics_file.close()

            
    return Alpha, Beta

if __name__ == '__main__':
    
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = np.float128(sys.argv[9])
    unique_y = np.array([0,1,2,3,4,5,6,7,8,9], dtype = np.float128)
    label_cnt = len(unique_y)

    Train_Data, Train_Label, Feature_Cnt =   file_reader(train_input, len(unique_y))
    Test_Data, Test_Label, Feature_Cnt2 = file_reader(test_input, len(unique_y))
    
    '''
    print(Test_Data[1])  
    print(Test_Label[1])
    print(Train_Data.shape)
    '''
    
        
    Alpha, Beta = SGD(Train_Data, Train_Label, Test_Data, Test_Label, Feature_Cnt,label_cnt,  hidden_units, init_flag, num_epoch, metrics_out)

    '''
    cross_entropy = 0.0
    for i in range(len(Train_Data)):
        NNF = NNForward(Train_Data[i], Train_Label[i], Alpha, Beta)
        cross_entropy = cross_entropy + NNF.getJ()
        
    avg_CE = cross_entropy/len(Train_Data)
    
    print(avg_CE)
        '''
    
    Predict(Train_Data, Train_Label,Alpha, Beta, metrics_out, train_out, "train" )
    Predict(Test_Data, Test_Label,Alpha, Beta, metrics_out, test_out,"test" )
    
    
    ##print(Alpha)
    ##print(Beta)
    
    
    
 
    
    '''
    print(Alpha.shape)
    print(Beta.shape)
    '''
    
    '''
    ##NN_Forward.setA()
    print(Beta.shape)
    print(Train_Data[1])
    print(Train_Label[0][1])
    print(Train_Data[1].shape)
    print(NN_Forward.getA())
    print(NN_Forward.getZ())
    print(NN_Forward.getYCap())
    print(NN_Forward.getJ())
    '''
    
    ##print(NN_Forward.getBeta())
    
    
    
    '''
    ##print(NN_Backward.getgB().shape)
    ##print(NN_Backward.getgBeta())
    print(NN_Backward.getgZ())
    print(NN_Backward.getgA())
    print(NN_Backward.getgAlpha().shape)
    print(NN_Backward.getgX().shape)
    '''

    
    