# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:39:41 2019

@author: cheth
"""


import sys
import numpy as np

if __name__ == '__main__':
    
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])
    
    def dictionary_reader(dictionary_file):
        dict_file = open(dictionary_file,"r")
        dict_file_list = dict_file.readlines()
        
        feature_dict = {}
        
        for i in range(len(dict_file_list)):
            
            feature_index = dict_file_list[i].split(' ')
            feature_dict[feature_index[0]] = feature_index[1].rstrip('\n')
            
        dict_file.close()
            
        return feature_dict
    
    
    def feature_reader_model1(input_file,dictionary):
        in_file = open(input_file,"r")
        in_file_list = in_file.readlines()
        
        
        label = []
        feature_index = []
        
        for i in range(len(in_file_list)):
            
            
            label_feature = in_file_list[i].split("\t")
            label.append(label_feature[0])
            ##feature_index.append(label_feature[1].rstrip('\n'))
            features = label_feature[1].rstrip('\n').split(" ")
            
            single_feature_index = []
            for f in features:
                
                try:
                    f_index = dictionary[f]
                    
                except:
                    continue
                
                try:
                    b=single_feature_index.index(f_index)
                    
                except ValueError:
                    single_feature_index.append(f_index)
                    
                else:
                    continue
        
        
            feature_index.append(single_feature_index)
            
        label_features = [label, feature_index]
        
        return label_features
    
    
    def feature_reader_model2(input_file,dictionary, threshold):
        in_file = open(input_file,"r")
        in_file_list = in_file.readlines()
        
        
        label = []
        feature_index = []
        
        for i in range(len(in_file_list)):
            
            
            label_feature = in_file_list[i].split("\t")
            label.append(label_feature[0])
            ##feature_index.append(label_feature[1].rstrip('\n'))
            features = label_feature[1].rstrip('\n').split(" ")
            
            single_feature_index = []
            single_feature_count = []
            for f in features:
                
                try:
                    f_index = dictionary[f]
                    
                except:
                    continue
                
                try:
                    ind=single_feature_index.index(f_index)
                    
                except ValueError:
                    single_feature_index.append(f_index)
                    single_feature_count.append(1)
                    
                else:
                    single_feature_count[ind] = single_feature_count[ind] + 1
                    
            
            single_feature_count = np.array(single_feature_count)
            single_feature_index = np.array(single_feature_index)
            
            
            remove_indices = np.nonzero(single_feature_count >= threshold)
            single_feature_index = np.delete(single_feature_index,remove_indices)
            
            single_feature_index = single_feature_index.tolist()
            
        
            feature_index.append(single_feature_index)
            
        label_features = [label, feature_index]
        
        return label_features
    
    def formatted_output(label_feature_index_list, formatted_out_file):
        
        observation_length = len(label_feature_index_list[0])
        
        outfile = open(formatted_out_file,"w")
        
        for i in range(observation_length):
            one_line = ""
            one_line = one_line + label_feature_index_list[0][i]
            
            for feature_ind in label_feature_index_list[1][i]:
                one_line= one_line + "\t" + feature_ind + ":1"
                
            outfile.write(one_line + "\n")
        
        outfile.close()
        
        return 0
        
    ########## Main #############################################################
    feature_dict = dictionary_reader(dict_input)
    
    
    if (feature_flag == 1):
        label_feature_model1_train = feature_reader_model1(train_input,feature_dict)
        label_feature_model1_valid = feature_reader_model1(validation_input,feature_dict)
        label_feature_model1_test = feature_reader_model1(test_input,feature_dict)
        
        formatted_output(label_feature_model1_train, formatted_train_out)
        formatted_output(label_feature_model1_valid, formatted_validation_out)
        formatted_output(label_feature_model1_test, formatted_test_out)
        
        
    else:
        label_feature_model2_train = feature_reader_model2(train_input,feature_dict, 4)
        label_feature_model2_valid = feature_reader_model2(validation_input,feature_dict, 4)
        label_feature_model2_test = feature_reader_model2(test_input,feature_dict, 4)
        
        formatted_output(label_feature_model2_train, formatted_train_out)
        formatted_output(label_feature_model2_valid, formatted_validation_out)
        formatted_output(label_feature_model2_test, formatted_test_out)
    

    
    
        
            
        
        
        