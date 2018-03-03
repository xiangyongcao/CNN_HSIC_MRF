# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 16:21:13 2017
@author: Xiangyong Cao
This code is modified based on https://github.com/KGPML/Hyperspectral
"""

import scipy.io
import numpy as np
from random import shuffle
import random
import scipy.ndimage
from skimage.util import pad
import os
import time
import pandas as pd
from utils import patch_size,convertToOneHot
import math
from sklearn.decomposition import PCA

flag_augment = True    # true if adopt data-augmentation strategy

start_time = time.time()

## Load data
DATA_PATH = os.path.join(os.getcwd(),"Data")
Data = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines.mat'))['indian_pines']
Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']

#Data = scipy.io.loadmat(os.path.join(DATA_PATH, 'PaviaU.mat'))['paviaU']
#Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'PaviaU_gt.mat'))['paviaU_gt']

#Data = scipy.io.loadmat(os.path.join(DATA_PATH, 'Simu_data.mat'))['Simu_data']
#Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'Simu_label.mat'))['Simu_label']

## Some constant parameters
Height, Width, Band = Data.shape[0], Data.shape[1], Data.shape[2]
Num_Classes = len(np.unique(Label))-1     # Simu: len(np.unique(Label))  


## Scale the HSI Data between [0,1]
Data = Data.astype(float)
for band in range(Band):
    Data[:,:,band] = (Data[:,:,band]-np.min(Data[:,:,band]))/(np.max(Data[:,:,band])-np.min(Data[:,:,band]))

#data_all = Data.transpose(2,0,1).transpose(0,2,1).reshape(Band,-1).transpose(1,0)
#n_components = 40
#pca = PCA(n_components=n_components)
#data_all = pca.fit_transform(data_all)
#Data = data_all.transpose(1,0).reshape(n_components,Width,Height).transpose(0,2,1).transpose(1,2,0)    
    
## padding the data beforehand
Height, Width, Band = Data.shape[0], Data.shape[1], Data.shape[2]
Data_Padding = np.zeros((Height+int(patch_size-1),Width+int(patch_size-1),Band))
for band in range(Band):
    Data_Padding[:,:,band] = pad(Data[:,:,band],int((patch_size-1)/2),'symmetric')
    
def Patch(height_index,width_index):
    """ function to extract patches from the orignal data """
    transpose_array = np.transpose(Data_Padding,(2,0,1))
    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    patch = transpose_array[:,height_slice, width_slice]
    return np.array(patch)   

## Collect patches for each class
Classes,Classes_Index = [],[]
for k in range(Num_Classes):
    Classes.append([])
    Classes_Index.append([])

All_Patches, All_Labels = [],[]   
for j in range(0,Width):
    for i in range(0,Height):
        curr_inp = Patch(i,j)
        curr_ind = j * Height + i
        curr_tar = Label[i,j]
        All_Patches.append(curr_inp)
        All_Labels.append(curr_tar)
        #Ignore patches with unknown landcover type for the central pixel
        if(curr_tar!=0): 
            Classes[curr_tar-1].append(curr_inp) 
            Classes_Index[curr_tar-1].append(curr_ind)
All_data = {}
All_data['patch'] = All_Patches
All_data['labels'] = All_Labels

Num_Each_Class=[]
for k in range(Num_Classes):
    Num_Each_Class.append(len(Classes[k]))
    

def DataDivide(Classes_k,Num_Train_Each_Class_k,Classes_Index_k):
    """ function to divide collected patches into training and test patches """
    np.random.seed(0)
    idx = np.random.choice(len(Classes_k), Num_Train_Each_Class_k, replace=False)
    train_patch = [Classes_k[i] for i in idx]
    train_index = [Classes_Index_k[i] for i in idx]
    idx_test = np.setdiff1d(range(len(Classes_k)),idx)
    test_patch = [Classes_k[i] for i in idx_test]
    test_index = [Classes_Index_k[i] for i in idx_test]
    return train_patch, test_patch, train_index, test_index   
       
## Make a train and test split 
# method 1: fixed proportion
train_frac = 0.05
Num_Train_Each_Class = [math.ceil(train_frac*x) for x in Num_Each_Class]
Num_Train_Each_Class = [int(x) for x in Num_Train_Each_Class] 

# method 2: a fixed number for each class                  
#Num_Train_Each_Class = [40] * Num_Classes

Num_Test_Each_Class = list(np.array(Num_Each_Class) - np.array(Num_Train_Each_Class))
Train_Patch, Train_Label, Test_Patch, Test_Label = [],[],[],[]
TestIndex, TrainIndex = [], []
for k in range(Num_Classes): 
    train_patch, test_patch, train_index, test_index  = DataDivide(Classes[k],Num_Train_Each_Class[k],
                                                     Classes_Index[k])
    TestIndex.extend(test_index)
    TrainIndex.extend(train_index)
    #Make training and test splits
    Train_Patch.append(train_patch)    # patches_of_current_class[:-test_split_size]
    Test_Patch.extend(test_patch)    # patches_of_current_class[-test_split_size:]
    Test_Label.extend(np.full(Num_Test_Each_Class[k], k, dtype=int))

Train_Label = []
for k in range(Num_Classes):
    Train_Label.append([k]*Num_Train_Each_Class[k])
    Resample_Num_Count = Num_Train_Each_Class
    
# Augment the data with random flipped and rotated patches
fixed_Train_Patch = Train_Patch    
if flag_augment:
    Resample_Num_Count = []
    times = 10    # can be tuned
    for k in range(Num_Classes):
        for l in range(times*Num_Train_Each_Class[k]):               
            if(len(Train_Patch[k])<times*Num_Train_Each_Class[k]):   
                num = random.randint(0,3)
                j = random.randint(0,Num_Train_Each_Class[k]-1)
                if num == 0 :
                    #Flip patch up-down
                    flipped_patch = np.flipud(fixed_Train_Patch[k][j]) 
                if num == 1 :
                    #Flip patch left-right
                    flipped_patch = np.fliplr(fixed_Train_Patch[k][j]) 
                if num == 2 :
                    #add gaussian noise
                    flipped_patch = Train_Patch[k][j] + np.random.normal(0,0.01,size = fixed_Train_Patch[k][j].shape) 
                if num == 3 :
                    #Rotate patch by a random angle
                    no = random.randrange(-180,180,30)
                    flipped_patch = scipy.ndimage.interpolation.rotate(fixed_Train_Patch[k][j], no,axes=(1, 0), 
                        reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=False)
                Train_Patch[k].append(flipped_patch)
                Train_Label[k].append(k)
        Resample_Num_Count.append(len(Train_Patch[k]))
                                
    OS_Aug_Num_Training_Each = []             
    for k in range(Num_Classes):
        OS_Aug_Num_Training_Each.append(len(Train_Label[k])) 

if flag_augment == False:
    OS_Aug_Num_Training_Each = Resample_Num_Count
            
# release list to elements             
Temp1,Temp2 = [],[]
for k in range(Num_Classes):
    Temp1.extend(Train_Patch[k])
    Temp2.extend(Train_Label[k])
Train_Patch = Temp1
Train_Label = Temp2

Train_Patch = np.array(Train_Patch)

# Convert the labels to One-Hot vector
Train_Label = np.array(Train_Label)
Test_Label = np.array(Test_Label)
Train_Label = convertToOneHot(Train_Label,num_classes=Num_Classes)
Test_Label = convertToOneHot(Test_Label,num_classes=Num_Classes)
                      
## Save the patches in segments
# Train Data
train_dict = {}
num_train = len(Train_Patch)
num_train_file = 10
num_each_file = int(num_train / num_train_file)
res_num = num_train - num_train_file * num_each_file
Num_Each_File = num_each_file * np.ones((1,num_train_file),dtype=int)
Num_Each_File = Num_Each_File[0]
Num_Each_File[num_train_file-1] = Num_Each_File[num_train_file-1] + res_num
start = 0
for i in range(num_train_file):
    file_name = 'Train_'+str(patch_size)+'_'+str(i+1)+'.mat'
    train_dict["train_patch"] = Train_Patch[start:start+Num_Each_File[i]]
    train_dict["train_labels"] = Train_Label[start:start+Num_Each_File[i],:]
    scipy.io.savemat(os.path.join(DATA_PATH, file_name),train_dict)
    start = start + Num_Each_File[i]
    
# Test Data
test_dict = {}
num_test = len(Test_Patch)
num_test_file = 10
num_each_file = int(num_test / num_test_file)
res_num = num_test - num_test_file * num_each_file
Num_Each_File = num_each_file * np.ones((1,num_test_file),dtype=int)
Num_Each_File = Num_Each_File[0]
Num_Each_File[num_test_file-1] = Num_Each_File[num_test_file-1] + res_num
start = 0
for i in range(num_test_file):
    file_name = 'Test_'+str(patch_size)+'_'+str(i+1)+'.mat'
    test_dict["test_patch"] = Test_Patch[start:start+Num_Each_File[i]]
    test_dict["test_labels"] = Test_Label[start:start+Num_Each_File[i],:]
    scipy.io.savemat(os.path.join(DATA_PATH, file_name),test_dict)
    start += Num_Each_File[i]


def add_DataSet(first,second,data_type):
    if data_type == 'train':
        temp_image = np.concatenate((first['train_patch'],second['train_patch']),axis=0)
        temp_labels = np.concatenate((first['train_labels'],second['train_labels']),axis=0)
        Combined_data = {}
        Combined_data['train_patch'] = temp_image
        Combined_data['train_labels'] = temp_labels
    if data_type == 'test':
        temp_image = np.concatenate((first['test_patch'],second['test_patch']),axis=0)
        temp_labels = np.concatenate((first['test_labels'],second['test_labels']),axis=0)
        Combined_data = {}
        Combined_data['test_patch'] = temp_image
        Combined_data['test_labels'] = temp_labels
    return Combined_data

def Prepare_data():
    """ functions to prepare Training and Testing data"""
    for i in range(num_train_file):
        file_name = 'Train_'+str(patch_size)+'_'+str(i+1)+'.mat'
        data_sets = scipy.io.loadmat(os.path.join(DATA_PATH, file_name))
        if(i==0):
            Training_data = data_sets
            continue
        else:
            Training_data = add_DataSet(Training_data,data_sets,'train')
            
    for i in range(num_test_file):
        file_name = 'Test_'+str(patch_size)+'_'+str(i+1)+'.mat'
        data_sets = scipy.io.loadmat(os.path.join(DATA_PATH, file_name))
        if(i==0):
            Test_data = data_sets
            continue
        else:
            Test_data = add_DataSet(Test_data,data_sets,'test')
    return Training_data, Test_data

Training_data, Test_data = Prepare_data()
  
print('Training Data:')
print(Training_data['train_patch'].shape)
print('Test Data:')
print(Test_data['test_patch'].shape)

# Data Summary
df = pd.DataFrame(np.random.randn(Num_Classes, 4),
                  columns=['Total', 'Training', 'OS&Aug', 'Testing'])
df['Total'] = Num_Each_Class
df['Training'] = Num_Train_Each_Class
df['OS&Aug'] = OS_Aug_Num_Training_Each
df['Testing'] = Num_Test_Each_Class
print("=======================================================================")
print("Data Summary")
print("=======================================================================")
print('The size of the original HSI data is (%d,%d,%d)'%(Height,Width,Band))
print('The size of Training data is (%d)'%(num_train))
print('The size of Test data is (%d)'%(num_test))
print('The size of each sample is (%d,%d,%d)'%(Band,patch_size,patch_size))
print('-----------------------------------------------------------------------')
print("The Data Division is")
print(df)
duration_time = time.time() - start_time
print("=======================================================================")
print('Data Preparation is Completed! (It takes %.5f seconds)'%(duration_time))
print("=======================================================================")

def load_index_data():
    data_path = os.getcwd()
    train_index = scipy.io.loadmat(os.path.join(data_path, 'TrainIndex.mat'))['TrainIndex']
    test_index = scipy.io.loadmat(os.path.join(data_path, 'TestIndex.mat'))['TestIndex']
    train_index = train_index[0]
    test_index = test_index[0]    

    TrainData = {}
    TrainData['train_patch']  = np.array([All_data['patch'][i] for i in train_index])
    TrainLabel = [All_data['labels'][i] for i in train_index]
    TrainLabel = np.array(TrainLabel)
    TrainLabel = convertToOneHot(TrainLabel-1,num_classes=Num_Classes)
    TrainData['train_labels']  = TrainLabel

    TestData = {}
    TestData['test_patch']  = np.array([All_data['patch'][i] for i in test_index])
    TestLabel = [All_data['labels'][i] for i in test_index]
    TestLabel = np.array(TestLabel)
    TestLabel = convertToOneHot(TestLabel-1,num_classes=Num_Classes)
    TestData['test_labels']  = TestLabel
    return TrainData, TestData, train_index, test_index