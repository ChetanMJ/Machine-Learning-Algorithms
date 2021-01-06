# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:13:11 2019

@author: cheth
"""

import sys
import math
import numpy as np

import time

if __name__ == '__main__':
    
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])
    
    
    
    def featureFile_to_vectors(input_feature_file, dictionary_file):
        dict_file = open(dictionary_file,"r")
        dict_file_list = dict_file.readlines()
        max_features = len(dict_file_list)
        
        weight_vector = np.zeros(max_features+1)
        
        weight_vector = weight_vector.tolist()
        
        dict_file.close()
        
        input_file = open(input_feature_file,"r")
        input_file_list = input_file.readlines()
        max_observations = len(input_file_list)
        
        feature_vector = []
        label_vector = []
        
        for i in range(max_observations):
            observation = input_file_list[i].split("\t")
            
            features = []
            
            for j in range(len(observation)):
                
                if j == 0:
                    label_vector.append(int(observation[j]))
                
                else:
                    index = observation[j].split(":")[0]
                    features.append(int(index))
            
            feature_vector.append(features)
            
          
        input_file.close()
        
        return feature_vector, label_vector, weight_vector
    
    
    def dotProduct(weight_vector, feature_vector_index):
        
        dot_prod = float(weight_vector[0]) * 1.0
        for feature_index  in feature_vector_index:
            dot_prod = dot_prod + (float(weight_vector[feature_index + 1]) * 1.0)
            
        return dot_prod
    
    def lr_SGD(weight_vector,feature_vector_index, labels_vector, epochs, learning_rate, formatted_validation_input):
        
        ##features_valid, labels_valid, w = featureFile_to_vectors(formatted_validation_input,dict_input)
        
        negative_log_likely = []
        for e in range(epochs):
            
            for i in range(len(feature_vector_index)):
                expn = math.exp(dotProduct(weight_vector,feature_vector_index[i]))
                
                weight_vector[0] = float(weight_vector[0]) + \
                                   ((learning_rate * 1.0) * \
                                   (labels_vector[i] - (expn / (1.0 + expn))))
                
                for feature_index in feature_vector_index[i]:
                    weight_vector[feature_index + 1] = float(weight_vector[feature_index + 1]) + \
                                                   ((learning_rate * 1.0) * \
                                                    (labels_vector[i] - (expn / (1.0 + expn))))
                                                           
            ##negative_log_likely.append(neg_log_likelihood(weight_vector,feature_vector_index,labels_vector))
                                                   
        
        return weight_vector, negative_log_likely
        
    def lr_predict(updated_weight_vector,feature_vector_index):
        
        y_cap = []
        
        for i in range(len(feature_vector_index)):
            P_Y = (1.0 / (1.0 + math.exp(-1.0 * dotProduct(updated_weight_vector,feature_vector_index[i]))))
            
            if P_Y >= 0.5 :
                y_cap.append(1)
            else:
                y_cap.append(0)
                
        return y_cap
    
    def neg_log_likelihood(updated_weight_vector, feature_index_vector, label_vector):
        
        neg_log_likely = 0
        N = len(feature_index_vector)
        
        for i in range(len(feature_index_vector)):
            neg_log_likely = neg_log_likely + ((-1.0 * int(label_vector[i]) * dotProduct(updated_weight_vector,feature_index_vector[i])) + \
                                                math.log(1 + math.exp(dotProduct(updated_weight_vector,feature_index_vector[i])))
                                                )
        
        return (neg_log_likely/N)
        
                                                
                                                
        
        
    ################# MAIN #########################################################################
    
    features, labels, weights = featureFile_to_vectors(formatted_train_input,dict_input)
    
    start = time.time()
    #### Train the weights #########
    updated_weight_vector, neg_log_likely = lr_SGD(weights, features, labels, num_epoch, 0.1, formatted_validation_input )
    
    end = time.time()
    
    print(end - start)
    
    print(neg_log_likely)
    
    #### Predict train data labels #####
    output = lr_predict(updated_weight_vector,features)
    
    #### Train error rate ###########
    train_error = 0
    
    predictTrainLabel_file= open(train_out,"w")
    
    for i in range(len(labels)):
        
        predictTrainLabel_file.write(str(output[i]) + "\n")
        
        if labels[i] <> output[i] :
            train_error = train_error + 1
            
        else:
            continue
     
    predictTrainLabel_file.close()
    
   
    
    ########  get the features and actual labels of test data ##############
    features_test, labels_test, weights = featureFile_to_vectors(formatted_test_input,dict_input)
    
    ######## predict test labels ###########################################
    output_test = lr_predict(updated_weight_vector,features_test)
    
    #### Test Error Rate ##################################################
    test_error = 0
    
    predictTestLabel_file= open(test_out,"w")
    
    for i in range(len(labels_test)):
        
        predictTestLabel_file.write(str(output_test[i]) + "\n")
        
        if labels_test[i] <> output_test[i] :
            test_error = test_error + 1
        else:
            continue
        
    predictTestLabel_file.close()
        
    
    ####### Metrics Output ###############################################
    
    MetricsOut_file= open(metrics_out,"w")
    
    MetricsOut_file.write("error(train): " + str(float(train_error)/float(len(labels))) + "\n")
    MetricsOut_file.write("error(test): " + str(float(test_error)/float(len(labels_test))) + "\n")
   
    MetricsOut_file.close()
    
    
    
    

    
    

    
    
                    
                    

        
        