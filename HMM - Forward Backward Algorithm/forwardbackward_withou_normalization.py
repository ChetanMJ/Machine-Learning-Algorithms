# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 02:44:07 2019

@author: cheth
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:23:51 2019

@author: chethan singh mysore jagadeesh (cmysorej)
"""

import sys
import numpy as np
import math

def input_reader(input_file,index_to_word,index_to_tag):
   in_file = open(input_file,"r")
   in_file_list = in_file.readlines()
        
   x=[]
   y=[]
   x_i=[]
   y_i = []
        
   for i in range(len(in_file_list)):
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
   index2_dict = {}
   
   for i in range(len(in_file_list)):
       index_dict[in_file_list[i].rstrip('\n')] = i ;
       index2_dict[i] = in_file_list[i].rstrip('\n');
       
   in_file.close()
   
   return index_dict,index2_dict

def alpha(x, index_to_tag, pi, ajk, bjk ):
    
    alpha_np = np.zeros([len(index_to_tag),len(x)], dtype = float)
    
    alpha_normalize_constant = np.zeros([len(x)], dtype = float)
    
    
    normalize_constant = sum(pi.transpose() * bjk[:,x[0]])
    
    alpha_np[:,0] = (pi.transpose() * bjk[:,x[0]]) ##/normalize_constant
    
    alpha_normalize_constant[0] = normalize_constant 
    
    
    for i in range(1,len(x)):
        
        normalize_constant = sum(bjk[:,x[i]] * np.dot(alpha_np[:,i-1].transpose(), ajk))
        alpha_np[:,i]  = (bjk[:,x[i]] * np.dot(alpha_np[:,i-1].transpose(), ajk)) ##/ normalize_constant
        alpha_normalize_constant[i] = normalize_constant 

    ##print(alpha_np)
    ##print(alpha_normalize_constant)
    return alpha_np  , alpha_normalize_constant


def beta(x,index_to_tag, pi, ajk, bjk ):
    
    beta_np = np.zeros([len(index_to_tag),len(x)], dtype = float)
    
    beta_np[:,len(x)-1] = 1.0 ##/ len(index_to_tag) 
    
    for i in range(len(x)-2, -1, -1):
        beta_np[:,i] = (np.dot(ajk,bjk[:,x[i+1]] * beta_np[:,i+1] )) ##/ sum(np.dot(ajk,bjk[:,x[i+1]] * beta_np[:,i+1]))
    
    return beta_np


def predict_file_write(x_i, y_cap,index_to_word,index_to_tag, predicted_file_out):
    
    output_string = ""
    for i in range(len(x_i)):
        word = index_to_word[x_i[i]]
        tag  = index_to_tag[y_cap[i]]
        output_string = output_string + word + "_" + tag + " "
        
    predicted_file_out.write(output_string.strip(" ") + "\n")



def forward_backward(x_i, y_i, index_to_word, index_to_tag, pi, ajk, bjk, predicted_file, metric_file):

    log_likelihood = 0
    accuracy_total = 0.0
    length_total = 0.0
    
    predicted_file_out= open(predicted_file,"w")
    
    for i in range(len(x_i)):
        alpha_np, alpha_normalize_constant = alpha(x_i[i], index_to_tag, pi, ajk, bjk )
        beta_np = beta(x_i[i], index_to_tag, pi, ajk, bjk )
        
        print(alpha_np)
        print(beta_np)
        
        '''
        try:
            if alpha_np[5][3] > beta_np[5][3]:
                print("alpha greater")
            elif alpha_np[5][3] == beta_np[5][3]:
                print("equal")
            elif alpha_np[5][3] < beta_np[5][3]:
                print("alpha less")
                
        except:
            print("Not found")
        '''
        
        alpha_beta = alpha_np * beta_np
        
        y_cap = np.argmax(alpha_beta, axis=0)
        
        predict_file_write(x_i[i], y_cap,index_to_word,index_to_tag, predicted_file_out)
        
        accuracy = 0.0
        length = 0.0
        
        for l in range(len(y_i[i])):
            
            length = length + 1.0
            if y_cap[l] == y_i[i][l]:
                accuracy = accuracy + 1.0
                
        accuracy_total = accuracy_total + (accuracy)
        length_total = length_total + length
 
        
        logy = 0.0
        for j in range(len(x_i[i])):
            logy = logy + math.log(alpha_normalize_constant[j])
        
        ##print(alpha_normalize_constant)
        ##print("ll = "+str(log_likelihood) )
        log_likelihood = log_likelihood + (math.log(sum(alpha_np[:,len(x_i[i])-1]))) ##+ math.log(y)
        ##log_likelihood = log_likelihood + logy
        ##print("i === " + str(i))
    
    metric_file_out= open(metric_file,"w")
    
    metric_file_out.write("Average Log-Likelihood: "+str(log_likelihood/len(x_i))+"\n")
    metric_file_out.write("Accuracy: "+str(float(accuracy_total)/float(length_total))+"\n")
    
    metric_file_out.close()
    predicted_file_out.close()
    

    
        
    





if __name__ == '__main__':
    
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]
    
    pi = np.loadtxt(hmmprior, dtype=float, delimiter=' ')
    ##print(pi)

    ajk = np.loadtxt(hmmtrans, dtype=float, delimiter=' ')
    ##print(ajk)
    
    bjk = np.loadtxt(hmmemit, dtype=float, delimiter=' ')
    ##print(bjk)
    
    index_to_word_dict, word_to_index_dict = index_reader(index_to_word)
    
    index_to_tag_dict, word_to_tag_dict = index_reader(index_to_tag)
    
    x_i, y_i = input_reader(test_input,index_to_word_dict,index_to_tag_dict)
    
    forward_backward(x_i, y_i, word_to_index_dict, word_to_tag_dict, pi, ajk, bjk, predicted_file, metric_file)
    
    