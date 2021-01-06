# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:17:04 2019

@author: cheth
"""

import sys
import math

if __name__ == '__main__':
    
    infile_train_name = sys.argv[1]
    infile_test_name = sys.argv[2]
    maximum_depth = int(sys.argv[3])
    labelfile_train_name = sys.argv[4]
    labelfile_test_name = sys.argv[5]
    metricsfile_name = sys.argv[6]
    
    ## Function to read file and convert it to matrix/list
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

    ##function get the unique list of values and frequency of those values in the input list
    def unique_value_count(input_list, refer_list = None):
        if refer_list == None:
            refer_list = list(input_list)
            
        y= list(refer_list)
        x= list(input_list)
        unique_values = list(set(y))
        unique_value_counts = []
        for i in range(len(unique_values)):
            unique_value_counts.append(x.count(unique_values[i]))
            
        return unique_values,unique_value_counts
          
    ##function to get the max vote value on the input list
    def max_vote_value(input_list,refer_list):
        unq_val, unq_val_cnt = unique_value_count(input_list,refer_list)
        
        max_vote_index = unq_val_cnt.index(max(unq_val_cnt))
        
        max_vote_val = unq_val[max_vote_index]
        
        return max_vote_val
        
    ## function to claculate entropy
    def entropy_marginal(input_list):
        unq_val, unq_val_cnt = unique_value_count(input_list)
        
        entp_mar = 0
        total_count = sum(unq_val_cnt)
        for i in range(len(unq_val)):
            entp_mar = entp_mar + ((-1)*((float(unq_val_cnt[i])/float(total_count)) * \
                              (math.log(float(unq_val_cnt[i])/float(total_count))/math.log(2))))
            
        return entp_mar
    
    ## function to calculate the conditional entropy for specific value
    def entropy_conditional_specific(lbl_list, attr_list, specific_value):
        sepecific_value_index_list=[ i for i,  z in enumerate(attr_list) if z==specific_value]
        specific_value_lbl_list = [z[1] for z in enumerate(lbl_list) if z[0] in sepecific_value_index_list]
        entrp_cond_specific = entropy_marginal(specific_value_lbl_list)
        return entrp_cond_specific
    
    ## function to calculate the conditional entropy
    def entropy_conditional(lbl_list,attr_list):
        attr_unq_val, attr_unq_val_cnt = unique_value_count(attr_list)
        
        entrp_cond_specific_list = []
        for i in range(len(attr_unq_val)):
            entrp_cond_specific_list.append(entropy_conditional_specific(lbl_list,attr_list,attr_unq_val[i]))
            
            
        entropy_conditional = 0
        for i in range(len(attr_unq_val)):
            entropy_conditional = entropy_conditional + ((float(attr_unq_val_cnt[i])/float(sum(attr_unq_val_cnt))) * entrp_cond_specific_list[i])
        
        return entropy_conditional
        
    ### function to calculat information gain        
    def information_gain(lbl_list,attr_list):
        entp_mar = entropy_marginal(lbl_list)
        entrp_cond = entropy_conditional(lbl_list,attr_list)
        information_gain = entp_mar - entrp_cond
        
        return information_gain
    
    ## decision tree class
    class DecisionTree :

        def __init__(self,MaxDepth, CurrentDepth, AttributeValueList,LabelValueList,AttributeNameList,LabelName):
            self.__MaxDepth = MaxDepth
            self.__CurrentDepth = CurrentDepth
            self.__LabelName = LabelName
            self.__AttributeValueList = AttributeValueList
            self.__LabelValueList = LabelValueList
            self.__AttributeNameList = AttributeNameList
            self.__NodeName = None
            self.__LeftValue = None
            self.__RightValue = None
            self.__LeftNode = None
            self.__RightNode = None

             
        def setMaxDepth(self, MaxDepth) :
            self.__MaxDepth = MaxDepth            
             
        def getMaxDepth(self) :
            return self.__MaxDepth            
             
        def setCurrentDepth(self, CurrentDepth) :
            self.__CurrentDepth = CurrentDepth            
             
        def getCurrentDepth(self) :
            return self.__CurrentDepth
         
        def setAttributeNameList(self, AttributeNameList) :
            self.__AttributeNameList = AttributeNameList            
             
        def getAttributeNameList(self) :
            return self.__AttributeNameList   

        def setLabelName(self, LabelName) :
            self.__LabelName = LabelName    
            
        def getLabelName(self) :
            return self.__LabelName

        def setAttributeValueList(self, AttributeValueList) :
            self.__AttributeValueList = AttributeValueList            
             
        def getAttributeValueList(self) :
            return self.__AttributeValueList
        
        def setLabelValueList(self, LabelValueList) :
            self.__LabelValueList = LabelValueList            
             
        def getLabelValueList(self) :
            return self.__LabelValueList
               
        def setNodeName(self, NodeName):
            self.__NodeName = NodeName
            
        def getNodeName(self):
            return self.__NodeName

        def setLeftValue(self, LeftValue):
            self.__LeftValue = LeftValue
            
        def getLeftValue(self):
            return self.__LeftValue
        
        def setRightValue(self, RightValue):
            self.__RightValue = RightValue
            
        def getRightValue(self):
            return self.__RightValue
        
        def setLeftNode(self, LeftNode):
            self.__LeftNode = LeftNode
            
        def getLeftNode(self):
            return self.__LeftNode
        
        def setRightNode(self, RightNode):
            self.__RightNode = RightNode
            
        def getRightNode(self):
            return self.__RightNode
        

    ### Function to train the decison tree based on the input  data 
    def TrainDecisonTree(MaxDepth,CurrentDepth,AttributeValueList,AttributeUniqueList,LabelValueList,LabelUniqueList,AttributeNameList,LabelName,side="None"):
    
        if len(AttributeValueList) == 0: return None
        if CurrentDepth > MaxDepth : return None
        

        ### get the information gain for the all the input attributes again the label list
        InformationGainList = []
       
        for i in range(len(AttributeNameList)):
            InformationGainList.append(information_gain(LabelValueList,AttributeValueList[i]))
            
        ## Indentify the attribute with maximum info again and indentify it as the node
        MaxInfoGainIndex = InformationGainList.index(max(InformationGainList))
        DecisonTreeTrain = DecisionTree(MaxDepth,CurrentDepth,AttributeValueList,LabelValueList,AttributeNameList,LabelName)
        DecisonTreeTrain.setNodeName(AttributeNameList[MaxInfoGainIndex])

 
        ### Indentify the value for left edge and right edge of the current node
        Att_unq_val, Att_unq_val_cnt = unique_value_count(AttributeValueList[MaxInfoGainIndex],AttributeUniqueList[MaxInfoGainIndex])
        DecisonTreeTrain.setLeftValue(Att_unq_val[0])
        DecisonTreeTrain.setRightValue(Att_unq_val[1])
        

        ## Based on the left edge value, slice the data from the input list to create the input data for the left node setting        
        LeftValueIndexList = [ i for i,  z in enumerate(AttributeValueList[MaxInfoGainIndex]) if z==DecisonTreeTrain.getLeftValue()]
        LeftLblValueList = [z[1] for z in enumerate(LabelValueList) if z[0] in LeftValueIndexList]
        LeftAttValueList = list(AttributeValueList)
        LeftAttValueList.pop(MaxInfoGainIndex)
        
        LeftAttNameList = list(AttributeNameList)
        LeftAttNameList.pop(MaxInfoGainIndex)
        LeftAttUnqList = list(AttributeUniqueList)
        LeftAttUnqList.pop(MaxInfoGainIndex)
        LeftLblUnqList = LabelUniqueList
        

        
        for i in range(len(LeftAttValueList)):
            LeftAttValueList[i] = [z[1] for z in enumerate(LeftAttValueList[i]) if z[0] in LeftValueIndexList]
            

        ## Based on the Right edge value, slice the data from the input list to create the input data for the right node setting        
        RightValueIndexList = [ i for i,  z in enumerate(AttributeValueList[MaxInfoGainIndex]) if z==DecisonTreeTrain.getRightValue()]
        RightLblValueList = [z[1] for z in enumerate(LabelValueList) if z[0] in RightValueIndexList]
        RightAttValueList = list(AttributeValueList)
        RightAttValueList.pop(MaxInfoGainIndex)
        
        RightAttNameList = list(AttributeNameList)
        RightAttNameList.pop(MaxInfoGainIndex)
        RightAttUnqList = list(AttributeUniqueList)
        RightAttUnqList.pop(MaxInfoGainIndex)
        RightLblUnqList = LabelUniqueList
        
        
        for i in range(len(RightAttValueList)):
            RightAttValueList[i] = [z[1] for z in enumerate(RightAttValueList[i]) if z[0] in RightValueIndexList]
            
        
        left_lbl_unq_val, left_lbl_unq_val_cnt = unique_value_count(LeftLblValueList,LeftLblUnqList)
        right_lbl_unq_val, right_lbl_unq_val_cnt = unique_value_count(RightLblValueList,RightLblUnqList)
        
        Left_Lbl_Min_Cnt = min(left_lbl_unq_val_cnt)
        Right_Lbl_Min_Cnt = min(right_lbl_unq_val_cnt)
        
        
        ##if the current depth has reached maxdepth or if left label list is perfectly classified the set the left leaf node based on max vote
        ##else set the left node 
        if (CurrentDepth == MaxDepth) or (Left_Lbl_Min_Cnt == 0) :
            print_string = ''
            
            for i in range(CurrentDepth):
                print_string = print_string + "| "
            
            print_string = print_string + DecisonTreeTrain.getNodeName() + "=" + DecisonTreeTrain.getLeftValue() +": ["+ str(left_lbl_unq_val_cnt[0]) +" "+ str(left_lbl_unq_val[0]) + "/" + str(left_lbl_unq_val_cnt[1]) +" "+ str(left_lbl_unq_val[1]) +"]"
            print(print_string)
            DecisonTreeTrain.setLeftNode(max_vote_value(LeftLblValueList,left_lbl_unq_val))
        else:
            NextDepth = CurrentDepth + 1
            
            print_string = ''
            
            for i in range(CurrentDepth):
                print_string = print_string + "| "
            
            print_string = print_string + DecisonTreeTrain.getNodeName() + "=" + DecisonTreeTrain.getLeftValue() +": [" + str(left_lbl_unq_val_cnt[0]) +" "+ str(left_lbl_unq_val[0]) + "/" + str(left_lbl_unq_val_cnt[1]) +" "+ str(left_lbl_unq_val[1]) +"]"
            print(print_string)
            
            DecisonTreeTrain.setLeftNode(TrainDecisonTree(MaxDepth,NextDepth,LeftAttValueList,LeftAttUnqList,LeftLblValueList,LeftLblUnqList,LeftAttNameList,LabelName,"Left"))

        ##if the current depth has reached maxdepth or if right label list is perfectly classified the set the right leaf node based on max vote
        ##else set the right node
        if (CurrentDepth == MaxDepth) or (Right_Lbl_Min_Cnt == 0) :
            print_string = ''
            
            for i in range(CurrentDepth):
                print_string = print_string + "| "
                
            print_string = print_string + DecisonTreeTrain.getNodeName() + "=" + DecisonTreeTrain.getRightValue() +": ["+ str(right_lbl_unq_val_cnt[0]) +" "+ str(right_lbl_unq_val[0]) + "/" + str(right_lbl_unq_val_cnt[1]) +" "+ str(right_lbl_unq_val[1]) +"]"
            print(print_string)
            
            DecisonTreeTrain.setRightNode(max_vote_value(RightLblValueList,right_lbl_unq_val))
            
        else:
            NextDepth = CurrentDepth + 1
            print_string = ''
            
            for i in range(CurrentDepth):
                print_string = print_string + "| "
                
            print_string = print_string + DecisonTreeTrain.getNodeName() + "=" + DecisonTreeTrain.getRightValue() +": ["+ str(right_lbl_unq_val_cnt[0]) +" "+ str(right_lbl_unq_val[0]) + "/" + str(right_lbl_unq_val_cnt[1]) +" "+ str(right_lbl_unq_val[1]) +"]"
            print(print_string)
                    
            DecisonTreeTrain.setRightNode(TrainDecisonTree(MaxDepth,NextDepth,RightAttValueList,RightAttUnqList,RightLblValueList,RightLblUnqList,RightAttNameList,LabelName,"Right"))
        
 
        return DecisonTreeTrain
        


    
    ### Function to traverse the decison tree based on the input data and predict the label    
    def DecisionTreeTraversal(Decision_Tree, row_data_list, atrribute_name_list):

        if isinstance(Decision_Tree, DecisionTree) == False :
            predicted_value = Decision_Tree
            return predicted_value
        else:
            CurrentNode = Decision_Tree.getNodeName()
            CurrentNodeIndex = atrribute_name_list.index(CurrentNode)
            CurrentNodeValue = row_data_list[CurrentNodeIndex]
        
        
        if Decision_Tree.getLeftValue() ==  CurrentNodeValue :
            if isinstance(Decision_Tree.getLeftNode(), DecisionTree) == False :
                predicted_value = str(Decision_Tree.getLeftNode())
                return predicted_value
            else:
                predicted_value = DecisionTreeTraversal(Decision_Tree.getLeftNode(),row_data_list, atrribute_name_list)
                    
        if Decision_Tree.getRightValue() ==  CurrentNodeValue :
            if isinstance(Decision_Tree.getRightNode(), DecisionTree) == False :
                predicted_value = str(Decision_Tree.getRightNode())
                return predicted_value
            else:
                predicted_value = DecisionTreeTraversal(Decision_Tree.getRightNode(),row_data_list, atrribute_name_list)
            
        return predicted_value
   
    
    ## function that loops through all records to call the decision tree traversal function to predict all labels
    def PredictLabels(Decision_Tree,data_file_name,Output_Label_File):
        data_file = open(data_file_name,"r")
        data_file_hdr_list = data_file.readlines()
        data_file_list = data_file_hdr_list[1:]
        data_header = data_file_hdr_list[0]
        atrribute_name_list= data_header.split(',')[0:-1]
        outfile = open(Output_Label_File,"w")
        error = 0
        for i in range(len(data_file_list)):
           
            line_list = data_file_list[i].split(',')
            Actual_label = line_list[len(line_list) - 1].rstrip('\n')
            Predicted_label = str(DecisionTreeTraversal(Decision_Tree,line_list, atrribute_name_list))

            if Actual_label == Predicted_label:
                error = error + 0
            else:
                error = error + 1
        
            outfile.write(str(Predicted_label) +"\n")
            
            
            
        error_rate = float(error) / float(len(data_file_list))
        outfile.close()
        data_file.close()
        return error_rate
    
    
    
    
    #### Main function###############################################################################################
    
    ## get the input data in matrix/list format
    att_value_list,att_unq_list,lbl_value_list,lbl_unq_list,att_name_list,lbl_name = infile_reader(infile_train_name)

    
    x,y = unique_value_count(lbl_value_list,lbl_unq_list)
    print("["+str(y[0])+" "+str(x[0])+"/"+str(y[1])+" "+str(x[1])+"]")
    
    if maximum_depth > len(att_name_list):
         maximum_depth = len(att_name_list)           
    
    if maximum_depth > 0 :
        TreeX = TrainDecisonTree(maximum_depth,1,att_value_list,att_unq_list,lbl_value_list,lbl_unq_list,att_name_list,lbl_name)
    else:
        TreeX = max_vote_value(lbl_value_list,lbl_unq_list)
        
    train_error_rate = PredictLabels(TreeX,infile_train_name,labelfile_train_name)
    test_error_rate = PredictLabels(TreeX,infile_test_name,labelfile_test_name)
    metrics_file = open(metricsfile_name,"w")
    metrics_file.write("error(train): "+str(train_error_rate) + "\n")
    metrics_file.write("error(test): "+str(test_error_rate) + "\n")
    metrics_file.close()
