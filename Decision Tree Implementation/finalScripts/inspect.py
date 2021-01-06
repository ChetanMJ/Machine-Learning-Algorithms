# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:14:11 2019

@author: cheth
"""


import sys
import os
import math

if __name__ == '__main__':
    
    infile_name = sys.argv[1]
    outfile_name = sys.argv[2]
    
    if not(os.path.isfile(infile_name)) :
        print("Input File does not exist")
        exit()
        
    outfile = open(outfile_name,"w")
    infile = open(infile_name,"r")

    infile_list = infile.readlines()[1:]
    
    data_set = []
    
    for line in infile_list:
        data_set.append(line.split(",")[-1].rstrip('\n'))
        
    class_list = list(set(data_set))
    
    total_count = len(data_set)

    class_occurance_count = list()

    entropy = 0

    for i in range(len(class_list)):
        class_occurance_count.append(data_set.count(class_list[i]))
        entropy = entropy + ((-1)*((float(class_occurance_count[i])/float(total_count)) * \
                              (math.log(float(class_occurance_count[i])/float(total_count))/math.log(2))))

    max_vote_index = class_occurance_count.index(max(class_occurance_count))

    non_max_votes = list(class_occurance_count)

    non_max_votes.pop(max_vote_index)

    max_vote_error = float(sum(non_max_votes)) / float(sum(class_occurance_count))

    outfile.write("entropy: "+str(entropy)+"\n")
    outfile.write("error: "+str(max_vote_error))

    outfile.close()
    infile.close()