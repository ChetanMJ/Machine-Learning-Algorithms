import sys
import os
import math

if __name__ == '__main__':
    
    infile_train_name = sys.argv[1]

    
    def infile_reader(data_file_name):
        data_file = open(data_file_name,"r")
        data_file_hdr_list = data_file.readlines()
        data_file_list = data_file_hdr_list[1:]
        data_header = data_file_hdr_list[0]
        atrribute_name_list= data_header.split(',')[0:-1]
        label_name = data_header.split(',')[-1].rstrip('\n')
        total_observations = len(data_file_list)
        total_atributes = len(data_file_list[0].split(',')) - 1
        atrribute_value_list=[['' for x in range(total_observations)] for y in range(total_atributes)]
        
        label_value_list=['' for y in range(total_observations)]
        
        
        for i in range(len(data_file_list)):
            
            line_list = data_file_list[i].split(',')
            
            for j in range(len(line_list)):
                
                if j == len(line_list) - 1 :
                    label_value_list[i] = line_list[j].rstrip('\n')
                
                else:
                    atrribute_value_list[j][i]=line_list[j]
                    
        data_file.close()
        attribute_unq_value_list = []
        for x in atrribute_value_list:
            attribute_unq_value_list.append(list(set(x)))
            
        lbl_unq_list = list(set(label_value_list))
        return atrribute_value_list,attribute_unq_value_list,label_value_list,lbl_unq_list,atrribute_name_list,label_name
        
    
    att_value_list,attribute_unq_value_list,lbl_value_list,lbl_unq_list,att_name_list,lbl_name = infile_reader(infile_train_name)
    print(att_name_list)
    print(attribute_unq_value_list)
    ##print(lbl_name)
    ##print(lbl_value_list)