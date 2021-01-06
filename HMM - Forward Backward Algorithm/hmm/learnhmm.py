# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:33:52 2019

@author: cheth
"""
import sys
import numpy as np

def input_reader(input_file,index_to_word,index_to_tag):
   in_file = open(input_file,"r")
   in_file_list = in_file.readlines()
        
   x=[]
   y=[]
   x_i=[]
   y_i = []
        
   for i in range(len(in_file_list)):
   ##for i in range(10):
       x_a = []
       y_a = []
       
       x_ai = []
       y_ai = []
       
            
       x_y = in_file_list[i].split(" ")
            
       for j in range(len(x_y)):
           xy = x_y[j].split("_")
           x_a.append(xy[0].rstrip('\n'))
           y_a.append(xy[1].rstrip('\n'))
           
           x_ai.append(index_to_word[xy[0].rstrip('\n')])
           y_ai.append(index_to_tag[xy[1].rstrip('\n')])
                
       x.append(x_a)
       y.append(y_a)
       x_i.append(x_ai)
       y_i.append(y_ai)
   
   in_file.close()        
   return np.array(x_i),np.array(y_i)


def index_reader(input_file):
   in_file = open(input_file,"r")
   in_file_list = in_file.readlines()
   
   index_dict = {}
   
   for i in range(len(in_file_list)):
       index_dict[in_file_list[i].rstrip('\n')] = i ;
    
   in_file.close()
   return index_dict
       
def pi_learn(x,y,index_to_word,index_to_tag):
    
   pi = np.ones([len(index_to_tag),1], dtype = float)
    
   for i in range(len(y)):
       index = y[i][0]
       pi[index] = pi[index] + 1
       
   pi = pi/sum(pi)
   
   return pi

def ajk_learn(x,y,index_to_word,index_to_tag):
    
    ajk = np.ones([len(index_to_tag),len(index_to_tag)], dtype = float)
        
    for i in range(len(y)):
        for j in range(len(y[i]) - 1):
            J = y[i][j]
            K = y[i][j + 1]
            
            ajk[J][K] = ajk[J][K] + 1
    
    for i in range(len(ajk)):
        ajk[i] = ajk[i]/sum(ajk[i])
         
    return ajk
     
def bjk_learn(x,y,index_to_word,index_to_tag):

    bjk = np.ones([len(index_to_tag), len(index_to_word)], dtype = float) 

    for i in range(len(y)):
        for j in range(len(y[i])):
            J = y[i][j]
            K = x[i][j]
            
            bjk[J][K] = bjk[J][K] + 1
            
    for i in range(len(bjk)):
        bjk[i] = bjk[i]/sum(bjk[i])
       
    return bjk

def output_write(matrix, output_file):
    op_file = open(output_file,"a")
    
    for i in range(len(matrix)):
        x = ""
        for j in range(len(matrix[i])):
            x = x + str(matrix[i][j]) + " "
            
        op_file.write(x.strip() + "\n")
    
    op_file.close()

            
            


if __name__ == '__main__':
    
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    index_to_word_dict = index_reader(index_to_word)
    index_to_tag_dict = index_reader(index_to_tag)
    x_i, y_i = input_reader(train_input,index_to_word_dict,index_to_tag_dict)
    ##print(index_to_word_dict)
    ##print(index_to_tag_dict)
    ##print(x_i)
    ##print(y_i) 
    
    pi = pi_learn(x_i, y_i, index_to_word_dict, index_to_tag_dict)
    ##print(pi)
    
    ajk = ajk_learn(x_i, y_i, index_to_word_dict, index_to_tag_dict)
    ##print(ajk)
        
    bjk = bjk_learn(x_i, y_i, index_to_word_dict, index_to_tag_dict)
    ##print(bjk)
    
    ##output_write(pi,hmmprior)
    ##output_write(ajk,hmmtrans)
    ##output_write(bjk,hmmemit)
    
    np.savetxt(hmmprior, pi, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='')
    np.savetxt(hmmtrans, ajk, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='')
    np.savetxt(hmmemit, bjk, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='')

               
               
               
               