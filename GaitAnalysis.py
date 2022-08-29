#!/usr/bin/env python
# coding: utf-8

# # CIE Group 4
# 
# ## Data Source: Smartphone 1, Accelearation
# ## Analysis: 2, 5 and 10 fold cross validation

# ## 

# ## Importing Libraries

# In[ ]:


# Necessary Libraries for Data Pre-Processing

import os
import pandas as pd
import numpy as np
import scipy 
from scipy import signal
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# Necessary Libraries for Neural Network Training

import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import KFold

from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras import metrics
import time


# ## 

# ## Defining Functions

# In[ ]:


def read_folders(path):
    
    """
    Function traverses through a folder and returns all the Accelerometer.csv files for normal & impaired walking
    key: subject[***]_[gait][**]
    value: numpy array of Accelerometer.csv file
    
    Parameters: Path of the data
    Output    : Dictionary with all the accelerometer files
    """
    
    d = {} # Creating empty dictionary

    # Looping through all the folders in data
    for root, dirs, files in os.walk(path , topdown=True):

        for name in files:
            file_path = os.path.join(root, name)  # Gives the full path to all csv files

            # Only consider files with the name "Accelerometer.csv" and ignore "Gyroscope.csv"
            if file_path[-17:]!="Accelerometer.csv":
                continue

            # Ignore data for upstairs and downstairs walking
            if "stairs" in root:
                continue

            # Generate subject number and type of data as key values
            rootsplit = root.split('\\')
            if "normal" in root or "impaired" in root or "Normal" in root or "Impaired" in root:
                key = rootsplit[-1]

            # Read the Accelerometer.csv file, convert to numpy array, and match with the key above
            d[key] = pd.DataFrame.to_numpy(pd.read_csv(file_path))
    
    return d          


# In[ ]:


def cleaning_data(orig_dict):
    
    """
    Criterias to ignore datasets from original dictionary
    
    Parameters: Original Dictionary
    Output    : Dictionary with cleaned data
    """
    
    clean_dictionary = orig_dict.copy()
    error = [] # List that contains all the erraneous datasets
    
    for key in orig_dict:
        
        mean_differences = np.mean(np.diff(orig_dict[key][:,0])) # Difference between consecutive times
        
        # Some data has 2 data recordings at same time. Ignoring that data
        if 0 in np.diff(orig_dict[key][:,0]):
            error.append(key)
            continue
        
        # Some data has all the recordings in 1 column. Ignoring that data.
        if np.asarray((orig_dict[key].shape)[1])<4:
            error.append(key)
            continue
        
        # Ideally the y axis should be in the gravity direction. If not, that data is ignored
        amean = [abs(np.mean(orig_dict[key][:,1])),abs(np.mean(orig_dict[key][:,2])),abs(np.mean(orig_dict[key][:,3]))]
        max_index = amean.index(max(amean))
        if max_index != 1:
            error.append(key)
            continue
        
        # If data recording is less that 25sec, Ignore that data.
        if (orig_dict[key][-1,0] - orig_dict[key][0,0]) < 25:
            error.append(key)
            continue
    
    # Remove the erraneous data entries in the dictionary named "clean_dict"
    error_array = np.unique(np.asarray(error))
    for keyerror in error_array:
        clean_dictionary.pop(keyerror) 
    
    return clean_dictionary


# In[ ]:


def get_frequency(orig_dict):
    
    """
    Obtains Frequency of each data
    
    Parameters: Original_Dict
    Output    : Frequency of each data
    """
    
    freq_dict={}
    
    # Frequency is not taken as 1/t[2]-t[1]. Since the consecutive differences are taken, freq=1/mean_timediff
    for key in orig_dict:        
        mean_differences = np.mean(np.diff(original_dict[key][:,0]))
        freq_dict[key] = 1/mean_differences         
        
    return freq_dict


# In[ ]:


def filtering(clean_dictionary, filtertype='butter', walk_freq = 3, order = 8):
    """
    Filtering Function: Butter or Savgol
    
    Parameters:
    walk_freq: Higher -> Noisy curve
    order: Higher -> Noisy curve
    
    Output   : Dictionary with filtered acceleration values
    """
    
    filtered_dict = {}

    for key in clean_dictionary:
        
        length = clean_dictionary[key].shape[0]
        time = clean_dictionary[key][:,0].reshape((length,1))
        filtered_dict[key] = np.zeros((length,4))
        
        
        if filtertype == 'savgol':
            for i in range(3):
                sig = clean_dictionary[key][:,i+1]
                if i == 1:
                    poly_order = 8
                else:
                    poly_order = 6
                sgfilter = signal.savgol_filter(sig, 101, poly_order)
                filtered_dict[key][:,0] = clean_dict[key][:,0]
                filtered_dict[key][:,i+1] = sgfilter
        else:
            for i in range(3):
                sig = clean_dictionary[key][:,i+1]
                sos = signal.butter(order, walk_freq, 'lp', fs = freq_dict[key], output='sos')
                butterfilter = signal.sosfiltfilt(sos, sig)
                filtered_dict[key][:,0] = clean_dict[key][:,0]
                filtered_dict[key][:,i+1] = butterfilter
    
    return filtered_dict


# In[ ]:


def absolute(filtered_dict):
    
    """
    Creates a dictionary with absolute values of acceleration = ((ax)^2 + (ay)^2 + (az)^2) ^ 1/2
    
    Parameters: Filtered Accelration
    Output    : Absolute Acceleration
    """    
    
    absolute_dict={}
    
    for key in filtered_dict:
        absolute_dict[key] = LA.norm(filtered_dict[key][:,1:4], axis=1)    
        
    return absolute_dict


# In[ ]:


def cut_data(absolute_dict, freq_dict, filtered_dict):    
    
    """
    Code to trim the irrelevant data at the start and end of the filtered sequence
    
    Parameters: absolute acceleration, frequency of data, filtered acceleration
    Output    : acceleration valuues in x,y,z after cutting the data
    """
    
    cut_dict = {}
    
    for key in absolute_dict:    
        
        # Finding the mean and max of the absolute acceleration
        mean = np.mean(absolute_dict[key]) 
        maxi = np.max(absolute_dict[key])
        
        #Finding peaks above the mean will identify the minor peaks as well. Finding mean of the peak will
        #give a correct height to identify the peaks
        pseudo_peaks, _ = find_peaks(absolute_dict[key], height = mean, distance = freq_dict[key]*0.5)
        pseudo_peaks_mean = np.mean(absolute_dict[key][pseudo_peaks])
        
        h = mean + (pseudo_peaks_mean-mean)*0.5
        
        # Finding peaks
        peaks, _ = find_peaks(absolute_dict[key], height=h, distance=freq_dict[key]*0.5)
        
        no_matter = 3    # To cut the data at the maximum difference between first 3 and last 3 peaks
        time_check_final = 2 # If the difference between 2 peaks in >2sec, cut the data
    
        diff_normalised = np.diff(peaks) / freq_dict[key]  # Peak differences in terms of time
        
        # Rear Index: Analyzing 3 peaks on the right ; Front Index: Analyzing 3 peaks on the left
        xcut_rear_index = diff_normalised.shape[0] - no_matter + np.argmax(diff_normalised[-no_matter:])
        xcut_front_index = np.argmax(diff_normalised[:no_matter]) + 1

        mean_time = np.mean(filtered_dict[key][peaks,0])   # To find the center point on X-axis (time axis)
        
        idx = (np.abs(filtered_dict[key][peaks,0] - mean_time)).argmin() # Finds the peak nearest to the mean_time

        remaining_mean = idx + 1
        countfr = 0  # Counter for front index
        countrr = 0  # Counter for rear index
        
        # Time difference check for the left part of the wave
        if any(x > time_check_final for x in diff_normalised[:remaining_mean]):
            rev_diff_rem_front = diff_normalised[:remaining_mean][::-1]
            for i in rev_diff_rem_front:       # Walk from the center to the left end of the wave
                countfr += 1
                if i>time_check_final:
                    xcut_front_index = remaining_mean - countfr + 1
                    break
        
        # Time difference check for the right part of the wave
        if any(x > time_check_final for x in diff_normalised[remaining_mean:]):
            diff_rem_rear = diff_normalised[remaining_mean:]
            for j in diff_rem_rear:
                countrr += 1
                if j>time_check_final:
                    xcut_rear_index = remaining_mean + countrr - 1
                    break

        cut_dict[key] = filtered_dict[key][peaks[xcut_front_index]:peaks[xcut_rear_index],:]
    
    return cut_dict


# In[ ]:


def rotate_data(cut_dictionary):
    
    """
    Rotates and transforms the data using PCA (Principal Component Analysis)
    
    Parameters: Cut Dictionary
    Output    : Rotated Dictionary
    """
    rotated_dict = {}
    for key in cut_dictionary:
        rotated_dict[key] = np.zeros((cut_dictionary[key].shape[0], 4))
        rotated = PCA(n_components = 3).fit_transform(cut_dictionary[key][:,1:4])
        rotated_dict[key][:,0] = cut_dictionary[key][:,0]
        rotated_dict[key][:,1:4] = rotated
        
    return rotated_dict


# In[ ]:


def rotate_axis(cut_dict):
    
    """
    Rotates and transforms the data using Axis Alignment
    
    Parameters: Cut Dictionary
    Output    : Rotated Dictionary
    """
    rotated_temp_dict={}
    
    for keys in cut_dict:
        rotated_temp_dict[keys]=cut_dict[keys]
        
        if (np.mean(cut_dict[keys][:,2],axis=0))<0:
            rotated_temp_dict[keys][:,1:3]=-1*rotated_temp_dict[keys][:,1:3]
        
    return rotated_temp_dict


# In[ ]:


def sample_data(cut_dict, freq_dict, samples=300):
    
    """
    Function for resampling of data
    
    Parameters: Cut Dictionary, Frequency of data, number of resampling points
    Output    : Dictionary of List of Arrays. Each array corresponds to 1 step
    """
    #sampled is a dictionary. Its values are a list containing numerous sampling arrays
    sampled_dict={}
    
    for keys in cut_dict:
        sampled_dict[keys]=[]
        
        acc_cut = cut_dict[keys][:,1:4]  # Extracts the acceleration values, and ignores time values
        res_acc = LA.norm(acc_cut, axis=1) # Absolute Acceleration
        avg = np.mean(res_acc)         # Average of the absolute acceleration
        dist = freq_dict[keys]*0.625    # Used in find_peaks
        
        pseud_peaks, _ = find_peaks(res_acc, height = avg, distance=dist)
        pseud_peaks_mean = np.mean(res_acc[pseud_peaks])
            
        h_new = avg + (pseud_peaks_mean-avg)*0.5
    
        index,_ = find_peaks(res_acc,height=h_new,distance=dist)   
        
        for i in range(index.shape[0]-1):
            sampled = np.zeros((3,samples))
            for k in range(3):
                segment = acc_cut[index[i]:index[i+1],k]
                sampled[k,:] = scipy.signal.resample(segment,samples)

            sampled_dict[keys].append(np.transpose(sampled))
        
    return sampled_dict


# In[ ]:


def remove_malicious_sequences_xyz(sampled_dict, stds):
    
    """
    Removes malicious sequence based on the standard deviation and x,y,z acceleration values
    
    Parameters: Sampled Dictionary
    Output    : Dictionary of Lists of Arrays. Each array corresponds to 1 step
    """
    neural_dict_xyz = {}
    counts = 0  # Returns the successfull steps
    
    for keys in sampled_dict:
        
        mean_sample_acc = np.zeros((sampled_dict[keys][0].shape[0],3))
        stddev_sample_acc = np.zeros((sampled_dict[keys][0].shape[0],3))
        
        for i in range(sampled_dict[keys][0].shape[0]):
            data_actual = np.zeros((len(sampled_dict[keys]),3))
            
            for j in range(len(sampled_dict[keys])):
                data_actual[j,0]=sampled_dict[keys][j][i,0]
                data_actual[j,1]=sampled_dict[keys][j][i,1]
                data_actual[j,2]=sampled_dict[keys][j][i,2]
                
            mean_sample_acc[i,0]=np.mean(data_actual[:,0])
            mean_sample_acc[i,1]=np.mean(data_actual[:,1])
            mean_sample_acc[i,2]=np.mean(data_actual[:,2])
            
            stddev_sample_acc[i,0]=np.std(data_actual[:,0])
            stddev_sample_acc[i,1]=np.std(data_actual[:,1])
            stddev_sample_acc[i,2]=np.std(data_actual[:,2])
            
        neural_dict_xyz[keys]=[]
    
        for k in sampled_dict[keys]:
            
            if np.all(np.abs(k[:,0]-mean_sample_acc[:,0])<=stds*np.mean(stddev_sample_acc[:,0]))            and np.all(np.abs(k[:,1]-mean_sample_acc[:,1])<=stds*np.mean(stddev_sample_acc[:,1]))             and np.all(np.abs(k[:,2]-mean_sample_acc[:,2])<=stds*np.mean(stddev_sample_acc[:,2])):
                neural_dict_xyz[keys].append(k)
                counts += 1
        
    return neural_dict_xyz, counts


# In[ ]:


def sample_clean_data(neural_dict_abs):
    
    """
    Removes subjects which did not generate any step data 
    
    Parameters: Neural Dictionary Absolute OR Neural Dictionary xyz
    Output    : Dictionary of Lists of Arrays. Each array corresponds to 1 step
    """
    sampled_cleaned_dict={}
    for keys in neural_dict_abs:
        if neural_dict_abs[keys]:
            sampled_cleaned_dict[keys] = neural_dict_abs[keys]
    return sampled_cleaned_dict
                


# In[ ]:


def normalize_data(sampled_sampled_dict):
   
    """
    Normalizes the data to input in the Neural Network
    
    Parameters: Sampled Clean Data
    Output    : Normalized and Sampled Clean Data.
    """
    normalized_dict = {}
    scaler = MinMaxScaler(feature_range=(-1,1))
    for key in sampled_sampled_dict:
        normalized_dict[key]=[]
        for lists in sampled_sampled_dict[key]:
            val = scaler.fit_transform(lists[:,0:3])
            val = val.reshape((val.shape[0]*3, 1), order ='F')
            val = val.T
            normalized_dict[key].append(val)
            
    return normalized_dict


# In[ ]:


def extract_label(normalized_dict):
    
    """
    Extracts labels from normalized data
    
    Parametrs: Normalized Dict
    Output   : Dictionary of Arrays. Each array corresponds to 1 subject
    """
    labeled_dict= {}
    
    for key, value in normalized_dict.items(): 
        if "impaired" in key:
            labeled_dict[key] = np.ones(len(value)) * 1
        if "normal" in key:
            labeled_dict[key] = np.ones(len(value)) * 0
    
    return labeled_dict


# In[ ]:


def generate_input(normalized_dict, labeled_dict, num_samples):
    
    """
    Generates Array of Training Data and Output
    
    Parameters: normalized_dict, labeled_dict
    Output    : X (Input Training Data) with dimensions = Steps x (3*resampling points), y (Array of Labels)
    """
    
    X = {key:np.array(value).reshape((len(value), 3 * num_samples)) for key, value in normalized_dict.items()}
    y = {key:value.reshape((value.shape[0], 1)) for key, value in labeled_dict.items()}
    
    return X, y


# ## 

# ## Data Pre-Processing

# In[ ]:


print("*************************************************************")
print("***********************PREPROCESSING*************************")
print("*************************************************************\n\n")


# In[ ]:


# Reading the original dictionary
original_dict=read_folders(r"D:\RWTH Notes\Semester 3\Computational Intelligence in Engineering\Project A\All Data\Smartphone1")
print("Length of Original Dictionary: {}".format(len(original_dict)))


# In[ ]:


# Cleaning the original dictionary
clean_dict = cleaning_data(original_dict)
print("Length of Clean Dictionary: {}".format(len(clean_dict)))


# In[ ]:


# Getting frequencies of the data
freq_dict = get_frequency(clean_dict)
print("Length of Frequency Dictionary: {}".format(len(freq_dict)))


# In[ ]:


# Filtering Operation
filtered_dict=filtering(clean_dict,'butter', walk_freq = 3, order = 8)
print("Length of Filtered Dictionary: {}".format(len(filtered_dict)))


# In[ ]:


# Taking absolute values of acceleration
absolute_dict = absolute(filtered_dict)
print("Length of Absolute Dictionary: {}".format(len(absolute_dict)))


# In[ ]:


# Cutting Dictionary
cut_dict= cut_data(absolute_dict, freq_dict, filtered_dict)
print("Length of Cut Dictionary: {}".format(len(cut_dict)))


# In[ ]:


# Rotated using Axis Alignment
# Malicious Sequences Removed Using xyz criteria

print("\n----------------------------------------------------------")
print("             Rotated Data Using Axis Alignment            ")
print("                 Sampled Using XYZ Criteria               ")
print("----------------------------------------------------------")

rotated_dict_axis_xyz = rotate_axis(cut_dict)
print("Length of Rotated Dictionary: {}".format(len(rotated_dict_axis_xyz)))

num_samples = 250
sampled_dict_axis_xyz = sample_data(rotated_dict_axis_xyz, freq_dict, samples = num_samples)
print("Length of Sampled Dictionary: {}".format(len(sampled_dict_axis_xyz)))

neural_dict_axis_xyz, steps_axis_xyz = remove_malicious_sequences_xyz(sampled_dict_axis_xyz, 1.5)
print("Total Number of Steps Extracted: {}".format(steps_axis_xyz))
print("Length of Neural Dictionary, after removing Malicious Sequences: {}".format(len(neural_dict_axis_xyz)))

sampled_cleaned_dict_axis_xyz = sample_clean_data(neural_dict_axis_xyz)
print("Length of Neural Dictionary, after removing Empty Lists: {}".format(len(sampled_cleaned_dict_axis_xyz)))

normalized_dict_axis_xyz = normalize_data(sampled_cleaned_dict_axis_xyz)
print("Length of Normalized Neural Dictionary: {}".format(len(normalized_dict_axis_xyz)))

labeled_dict_axis_xyz = extract_label(normalized_dict_axis_xyz)
print("Length of Labeled Dictionary: {}".format(len(labeled_dict_axis_xyz)))


# In[ ]:


# Generate Input Arrays which are input to the Neural Network

X, y = generate_input(normalized_dict_axis_xyz, labeled_dict_axis_xyz, num_samples)

print("\n----------------------------------------------------------")
print("                     Input Data Summary                   ")
print("----------------------------------------------------------") 
print("Size of Input Array  : {} x {}".format(steps_axis_xyz, 3*num_samples))
print("Size of Labeled Array: {} x {}\n\n\n\n".format(steps_axis_xyz, 1))


# ##  
# 

#  ## Neural Network Model

# In[ ]:


print("*************************************************************")
print("***********************MODEL SUMMARY*************************")
print("*************************************************************\n\n")


# In[ ]:


# Set the hyperparameters
learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=1000,decay_rate=0.9)
folds = [2, 5, 10]
num_classes = 1


# In[ ]:


# Building the model using Relu

relu_model = Sequential()  
relu_model.add(Input(shape = (3 * num_samples,)))

regularizer = 0.1
# Number of hidden layers kept variable
# relu_model.add(Dense(2048, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(regularizer)))
# relu_model.add(Dense(1024, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(regularizer)))
# relu_model.add(Dense(512, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(regularizer)))
relu_model.add(Dense(256, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(regularizer)))
relu_model.add(Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(regularizer)))
relu_model.add(Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(regularizer)))
relu_model.add(Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(regularizer)))
relu_model.add(Dense(16, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(regularizer)))
relu_model.add(Dense(8, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(regularizer)))
relu_model.add(Dense(4, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(regularizer)))
# relu_model.add(Dense(2, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(regularizer)))
    
relu_model.add(Dense(num_classes, activation = 'sigmoid'))

# Manually define the optimizer in case you wish to change the learning rate/Optimizer
opt = Adam(learning_rate = learn_rate)
relu_model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = [metrics.BinaryAccuracy(), 'mse'])
print("\n----------------------------------------------------------")
print("                        ReLU Model                        ")
print("----------------------------------------------------------") 
print(relu_model.summary())


# ### 

# ## Training

# In[ ]:


print("\n\n\n\n\n*************************************************************")
print("**************************TRAINING***************************")
print("*************************************************************\n\n")


# In[ ]:


start_time = time.time()

relu_history = []

for num_folds in folds:
    
    kf = KFold(num_folds, shuffle = True, random_state=28)    

    X_names = []  # List to store subject names
    for key in X:
        X_names.append(key)
    
    iter_flag = 0
    
    for train, test in kf.split(X_names):
        
        iter_flag += 1
        x_train = np.array([]).reshape((0, 3 * num_samples))
        y_train = np.array([]).reshape((0, 1))
        x_val = np.array([]).reshape((0, 3 * num_samples))
        y_val = np.array([]).reshape((0, 1))

        for i in train:
            x_train = np.vstack((x_train, X[X_names[i]]))
            y_train = np.vstack((y_train, y[X_names[i]]))

        for i in test:
            x_val = np.vstack((x_val, X[X_names[i]]))
            y_val = np.vstack((y_val, y[X_names[i]]))
        
        print('\n-------------------------------------------')
        print('              Number of Folds: {}            '.format(num_folds))
        print('                 Iteration: {}               '.format(iter_flag))
        print('---------------------------------------------')
        print("Number of Total Subjects:   {}".format(len(X_names)))
        print("Subjects in Training:       {}".format(len(train)))
        print("Subjects in Validation:     {}".format(len(test)))
        print("Number of Total Subjects:   {}".format(len(X_names)))
        print("Steps in Input Training:    {}".format(x_train.shape))
        print("Steps in Output Training:   {}".format(y_train.shape))
        print("Steps in Input Validation:  {}".format(x_val.shape))
        print("Steps in Output Validation: {}\n\n".format(y_val.shape))

        relu_history.append(relu_model.fit(x_train, y_train, validation_data=(x_val,y_val), verbose = 1, epochs=50, batch_size=32))
        
    print("\n\n############################################################################\n\n")
end_time = time.time()

print("\n\n-------------Total Training Time: {}s-------------\n\n".format(end_time - start_time))


# In[ ]:


print("\n\n\n\n\n******************************************************************")
print("*******************  Done Training. Goodbye!  ********************")
print("******************************************************************")

