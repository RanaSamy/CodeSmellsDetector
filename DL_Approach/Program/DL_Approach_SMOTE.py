#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from sklearn.metrics import confusion_matrix
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt
import numpy as np
import gc
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
import seaborn as sns
from time import time
from collections import Counter
from imblearn.over_sampling import SMOTE
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras import regularizers


# In[7]:


pip install h5py==2.9


# ### Get Data

# In[2]:


class Input_data:
    def __init__(self, train_data, train_labels, eval_data, eval_labels, max_input_length):
        self.train_data = train_data
        self.train_labels = train_labels
        self.eval_data = eval_data
        self.eval_labels = eval_labels
        self.max_input_length = max_input_length

############################################################################################################

def compute_max(arr, dim="width", z=2):
    mn = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0)
    final_list = [x for x in arr if (x <= mn + z * sd)]  # upper outliers removed
    rmn2 = len(arr) - len(final_list)
    print('{} array size '.format(dim) + str(len(arr)))
    print('min {} '.format(dim) + str(min(arr)))
    print('max {} '.format(dim) + str(max(arr)))
    print('mean {} '.format(dim) + str(np.nanmean(arr)))
    print('standard deviation ' + str(np.std(arr)))
    print('median {} '.format(dim) + str(np.nanmedian(arr)))
    print('number of upper outliers removed ' + str(rmn2))
    print('max {} excluding upper outliers '.format(dim) + str(max(final_list)))
    return max(final_list)


############################################################################################################

def _get_outlier_threshold(path):
    lengths = []
    for root, dirs, files in os.walk(path):
        for f in files:
            filepath = os.path.join(root, f)
            with open(filepath, "r", errors='ignore') as file:
                for line in file:
                    #input_str = line.replace("\t", " ")
                    #input_str = input_str.replace("\n", " ")
                    np_arr = np.fromstring(line, dtype=np.int32, sep=" ")
                    if -1 in np_arr:
                        print("-1 happened")
                    cur_width = len(np_arr)
                    lengths.append(cur_width)
    return compute_max(lengths)


############################################################################################################

def get_outlier_threshold(path, z=1, is_c2v=False):
    len1 = _get_outlier_threshold(os.path.join(path, "Positive"))
    len2 = _get_outlier_threshold(os.path.join(path, "Negative"))
    if len1 > len2:
        return len1
    else:
        return len2


############################################################################################################

def get_data_files(path, max_len):
    input = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r',
                  errors='ignore') as file_read:
            for line in file_read:
                #input_str = line.replace("\t", " ")
                #input_str = input_str.replace("\n", " ")
                arr = np.fromstring(line, dtype=np.int32, sep=" ", count=max_len)
                arr_size = len(np.fromstring(line, dtype=np.int32, sep=" "))
                if arr_size <= max_len:
                    arr[arr_size:max_len] = 0
                    input.append(arr)
    return input


############################################################################################################

def get_data(data_path, train_validate_ratio):
    max_input_length = get_outlier_threshold(data_path)

    # Positive cases
    folder_path = os.path.join(data_path, "Positive")
    pos_data_arr = get_data_files(folder_path, max_input_length)
    shuffle(pos_data_arr)
    total_positive_cases = len(pos_data_arr)
    total_positive_labels = np.ones(shape=total_positive_cases, dtype=np.int8)
    print("total positive cases: " + str(total_positive_cases))
    
    # Negative cases
    folder_path = os.path.join(data_path, "Negative")
    neg_data_arr = get_data_files(folder_path, max_input_length)
    shuffle(neg_data_arr)
    total_negative_cases = len(neg_data_arr)
    total_negative_labels = np.zeros(shape=total_negative_cases, dtype=np.int8)
    print("total negative cases: " + str(total_positive_cases))
    
    # Merge positive and Negative cases
    all_data = []
    all_data.extend(pos_data_arr[0:total_positive_cases])
    all_data.extend(neg_data_arr[0:total_negative_cases])
    all_data_arr = np.array(all_data, dtype=np.int32)
    all_data_labels=[]
    all_data_labels.extend(total_positive_labels[0:total_positive_cases])
    all_data_labels.extend(total_negative_labels[0:total_negative_cases])
    all_data = all_data_arr.reshape((len(all_data_labels), max_input_length))
    print("All data: " + str(all_data.shape))
    
    # Apply SMOTE 
    smt = SMOTE(sampling_strategy=0.2)
    all_data,all_data_labels = smt.fit_resample(all_data, all_data_labels)
    print("All data: " + str(all_data.shape))
    
    # Split the new data
    SMOTE_positive_data=[]
    SMOTE_negative_data=[]
    all_data_number= len(all_data_labels)
    for i in range(0, all_data_number):
        if all_data_labels[i] == 1:
            SMOTE_positive_data.append(all_data[i])
        else:
            SMOTE_negative_data.append(all_data[i])
            
    SMOTE_total_positive_cases = len(SMOTE_positive_data)
    SMOTE_total_negative_cases = len(SMOTE_negative_data)
    print("Positive data after applying SMOTE: "+ str(SMOTE_total_positive_cases))
    print("Negative data after applying SMOTE: "+ str(SMOTE_total_negative_cases))
    
    # Split data into training and test portions
    total_training_positive_cases = int(train_validate_ratio * SMOTE_total_positive_cases)
    total_eval_positive_cases = int(total_positive_cases - total_training_positive_cases)

    total_training_negative_cases = int(train_validate_ratio * SMOTE_total_negative_cases)
    total_eval_negative_cases = int(total_negative_cases - total_training_negative_cases)
  

    training_data = []
    training_data.extend(SMOTE_positive_data[0:total_training_positive_cases])
    training_data.extend(SMOTE_negative_data[0:total_training_negative_cases])
    training_data_arr = np.array(training_data, dtype=np.int32)

    training_labels = np.empty(shape=[len(training_data_arr)], dtype=np.int8)
    training_labels[0:total_training_positive_cases] = 1
    training_labels[total_training_positive_cases:len(training_data_arr)] = 0

    eval_data = []
    eval_data.extend(SMOTE_positive_data[len(pos_data_arr) - total_eval_positive_cases:])
    eval_data.extend(neg_data_arr[len(neg_data_arr) - total_eval_negative_cases:])
    eval_data_arr = np.array(eval_data, dtype=np.int32)

    eval_labels = np.empty(shape=[len(eval_data_arr)], dtype=np.int8)
    eval_labels[0:total_eval_positive_cases] = 1
    eval_labels[total_eval_positive_cases:] = 0
    
    training_data = training_data_arr.reshape((len(training_data_arr), max_input_length))
    eval_data = eval_data_arr.reshape((len(eval_labels), max_input_length))
    training_data,training_labels = shuffle(training_data,training_labels)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)

    return training_data, training_labels, eval_data, eval_labels, max_input_length

############################################################################################################

def get_all_data(data_path):
    print("reading data...")

    train_data, train_labels, eval_data, eval_labels, max_input_length =         get_data(data_path, train_validate_ratio=TRAIN_VALIDATE_RATIO )
    print("nan count: " + str(np.count_nonzero(np.isnan(train_data))))
    print("train_data: " + str(len(train_data)))
    print("train_data shape: " + str(train_data.shape))
    print("eval_data: " + str(len(eval_data)))
    print("eval_labels: " + str(len(eval_labels)))
    print("reading data... done.")
    input_data = Input_data(train_data, train_labels, eval_data, eval_labels, max_input_length)
    return input_data


# In[ ]:





# In[3]:


TRAIN_VALIDATE_RATIO = 0.7
OUT_FOLDER = "D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Result"
data_path = "D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\ExtractedData\\TokenizedSamples"
smell = "FeatureEnvy"
input_data = get_all_data(data_path)


# In[ ]:





# In[4]:


def write_result(file, str):
    f = open(file, "a+")
    f.write(str)
    f.close()
    
def get_out_file(smell):
    now = datetime.datetime.now()
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    return os.path.join(OUT_FOLDER, "ae_" + smell + "_"
                        + str(now.strftime("%d%m%Y_%H%M") + ".csv"))

############################################################################################################

def find_metrics(error_df, threshold):
    y_pred = [1 if e > threshold else 0 for e in error_df.Reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.True_class, y_pred)
    precision, recall, f1 = compute_metrics(conf_matrix)
    return threshold, precision, recall, f1

############################################################################################################

def compute_metrics(conf_matrix):
    precision = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])
    recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
    f1 = (2 * precision * recall) / (precision + recall)
    print("precision: " + str(precision) + ", recall: " + str(recall) + ", f1: " + str(f1))
    return precision, recall, f1

############################################################################################################

def get_predicted_y(prob, threshold):
    out_arr = np.empty(len(prob), dtype=np.int32)
    for i in range(0, len(prob)):
        if prob[i] > threshold:
            out_arr[i] = 1
        else:
            out_arr[i] = 0
    return out_arr


# In[ ]:





# In[5]:


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def get_all_metrics(model, eval_data, eval_labels, pred_labels):
    fpr, tpr, thresholds_keras = roc_curve(eval_labels, pred_labels)
    auc_ = auc(fpr, tpr)
    print("auc_keras:" + str(auc_))

    score = model.evaluate(eval_data, eval_labels, verbose=0)
    print("Test accuracy: " + str(score[1]))

    precision = precision_score(eval_labels, pred_labels)
    print('Precision score: {0:0.2f}'.format(precision))

    recall = recall_score(eval_labels, pred_labels)
    print('Recall score: {0:0.2f}'.format(recall))

    f1 = f1_score(eval_labels, pred_labels)
    print('F1 score: {0:0.2f}'.format(f1))

    average_precision = average_precision_score(eval_labels, pred_labels)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    return auc_, score[1], precision, recall, f1, average_precision, fpr, tpr

############################################################################################################

def get_all_metrics_(eval_labels, pred_labels):
    fpr, tpr, thresholds_keras = roc_curve(eval_labels, pred_labels)
    auc_ = auc(fpr, tpr)
    print("auc_keras:" + str(auc_))

    precision = precision_score(eval_labels, pred_labels)
    print('Precision score: {0:0.2f}'.format(precision))

    recall = recall_score(eval_labels, pred_labels)
    print('Recall score: {0:0.2f}'.format(recall))

    f1 = f1_score(eval_labels, pred_labels)
    print('F1 score: {0:0.2f}'.format(f1))

    average_precision = average_precision_score(eval_labels, pred_labels)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    return auc_, precision, recall, f1, average_precision, fpr, tpr


# In[ ]:





# In[6]:


from inspect import signature
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def save_roc_curve(eval_labels, pred_labels, file_name):
    fig = plt.figure()
    lw = 2
    fpr, tpr, _=roc_curve(eval_labels, pred_labels)
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)')
    plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    fig.savefig(file_name)

def save_precision_recall_curve(eval_labels, pred_labels, file_name):
    fig = plt.figure()
    precision, recall, _ = precision_recall_curve(eval_labels, pred_labels)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    fig.savefig(file_name)


# In[ ]:





# # AutoEncoders
# 

# In[23]:


from sklearn.model_selection import cross_val_score

# Binary Classification with Sonar Dataset: Baseline
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, shuffle=True)

def autoencoder_dense(data, smell, no_of_layers=1, encoding_dimension=32, epochs=10, with_bottleneck=True, threshold=400000):
    input_layer = Input(shape=(data.max_input_length,))
    prev_layer = input_layer
    for i in range(no_of_layers):
        encoder = Dense(int(encoding_dimension / pow(2, i)), activation="relu",
                        activity_regularizer=regularizers.l1(10e-3))(prev_layer)
        prev_layer = encoder
    # bottleneck
    if with_bottleneck:
        prev_layer = Dense(int(encoding_dimension / pow(2, no_of_layers)), activation="relu")(prev_layer)
    for j in range(no_of_layers - 1, -1, -1):
        decoder = Dense(int(encoding_dimension / pow(2, j)), activation='relu')(prev_layer)
        prev_layer = decoder
    prev_layer = Dense(data.max_input_length, activation='relu')(prev_layer)
    prev_layer = Dense(1, activation='sigmoid')(prev_layer)
    autoencoder = Model(inputs=input_layer, outputs=prev_layer)

    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
    autoencoder.summary()
    history = autoencoder.fit(data.train_data,
                              data.train_labels,
                              epochs=epochs,
                              batch_size=32,
                              verbose=1,
                              shuffle=True).history

    plt.plot(history['loss'])
#     plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    filename3= 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Result\\AutoEncoder\\smote_ae_256'+'\\model.png' 
    #SVG(model_to_dot(autoencoder).create(prog='dot', format='svg'))
    plot_model(autoencoder, to_file=filename3, show_shapes=True)
    
    prob = autoencoder.predict(data.eval_data)
    
    y_pred = get_predicted_y(prob, 0.5) 
    
    
    auc, accuracy, precision, recall, f1, average_precision, fpr, tpr = get_all_metrics(autoencoder, data.eval_data, data.eval_labels, y_pred)
    
    filename1= 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Result\\AutoEncoder\\smote_ae_256\\roc_curve.png' 
    filename2= 'D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Result\\AutoEncoder\\smote_ae_256\\precision_recall_curve.png'           
    save_roc_curve(data.eval_labels, y_pred, filename1)
    save_precision_recall_curve(data.eval_labels, y_pred,filename2)
    
    #predictions = autoencoder.predict(data.eval_data)
    mse = np.mean(np.power(data.eval_data - prob, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': data.eval_labels})
    
    save_roc_curve(data.eval_labels, y_pred, filename1)
    save_precision_recall_curve(data.eval_labels, y_pred,filename2)
    
    return find_metrics(error_df, threshold)

############################################################################################################

def autoencoder(smell, input_data, layer, epochs=20, encoding=1024, bottleneck=True, threshold=400000):
    outfile = get_out_file(smell)
    write_result(outfile,
                 "Encoding_dim,threshold,epoch,bottleneck,layer,precision,recall,f1,time\n")
    #start_time = time.time()
    try:
        optimal_threshold, max_pr, max_re, max_f1 = autoencoder_dense(input_data, smell, no_of_layers=layer,
                                                                      epochs=epochs,
                                                                      encoding_dimension=encoding,
                                                                      with_bottleneck=bottleneck,
                                                                      threshold=threshold)
    except ValueError as error:
        print(error)
    #end_time = time.time()
    #time_taken = end_time - start_time
    #write_result(outfile,str(encoding) + "," + str(optimal_threshold) + "," + str(epochs) + "," + str(bottleneck) + "," + str(layer) + "," +str(max_pr) + "," + str(max_re) + "," + str(max_f1) + "," + str(time_taken) + "\n")


# In[24]:


autoencoder("FeatureEnvy", input_data=input_data, layer=1,epochs=5, encoding=32, bottleneck=True, threshold=319000)


# In[ ]:





# In[ ]:





# # CNN

# In[21]:


def CNN(data, batch_size,epochs_number,emb_output, out_folder):
    
    
    max_features = int(max(np.max(data.train_data), np.max(data.eval_data)))
    min_features= int(min(np.min(data.train_data), np.min(data.eval_data)))
    print("max_features: ",max_features)
    print("min_features: ",min_features)
    
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(data.max_input_length,)))
    model.add(tf.keras.layers.Embedding(input_dim=max_features + 1,output_dim=emb_output,mask_zero=True))
    #model.add(tf.keras.layers.Conv1D(32, 11, activation='relu'))
    model.add(tf.keras.layers.Conv1D(16, 5, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D((5), strides=2))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1,mode='auto')
    
    history= model.fit(data.train_data, data.train_labels, validation_split=0.2, epochs=epochs_number,batch_size=batch_size, verbose=1, shuffle=True).history
    
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    prob = model.predict(data.eval_data)
    y_pred = get_predicted_y(prob, 0.5)
    try:
        auc, accuracy, precision, recall, f1, average_precision, fpr, tpr = get_all_metrics(model, data.eval_data, data.eval_labels, y_pred)
    except Exception as e:
        print(e)
    
    filename1= out_folder+'\\roc_curve.png' 
    filename2= out_folder+'\\precision_recall_curve.png' 
    filename3= out_folder+'\\model.png' 
    
    save_roc_curve(data.eval_labels, y_pred, filename1)
    save_precision_recall_curve(data.eval_labels, y_pred,filename2)
    plot_model(model, to_file=filename3, show_shapes=True)
    #model.save(out_folder+'\\weights.hdf5')
    
    return auc, accuracy, precision, recall, f1, average_precision


# In[22]:


out_folder = "D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Result\\CNN\\smote_cnn_256"
CNN(input_data,256,5,16,out_folder)


# In[ ]:





# # LSTM

# In[42]:


def LSTM(data, batch_size,epochs_number,emb_output, out_folder):
    
    
    max_features = int(max(np.max(data.train_data), np.max(data.eval_data)))
    min_features= int(min(np.min(data.train_data), np.min(data.eval_data)))
    
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(data.max_input_length,)))
    model.add(tf.keras.layers.Embedding(input_dim=max_features + 1,output_dim=emb_output,mask_zero=True))
    #model.add(tf.keras.layers.LSTM(64, return_sequences=True, recurrent_dropout=0.1, dropout=0.1))
    model.add(tf.keras.layers.LSTM(32, return_sequences=True, recurrent_dropout=0.1, dropout=0.1))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1,mode='auto')
    
    history= model.fit(data.train_data, data.train_labels, validation_split=0.2, epochs=epochs_number,batch_size=batch_size, verbose=1, shuffle=True).history
    
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    prob = model.predict(data.eval_data)
    y_pred = get_predicted_y(prob, 0.5)
    try:
        auc, accuracy, precision, recall, f1, average_precision, fpr, tpr = get_all_metrics(model, data.eval_data, data.eval_labels, y_pred)
    except Exception as e:
        print(e)
    
    filename1= out_folder+'\\roc_curve.png' 
    filename2= out_folder+'\\precision_recall_curve.png' 
    filename3= out_folder+'\\model.png' 
    
    save_roc_curve(data.eval_labels, y_pred, filename1)
    save_precision_recall_curve(data.eval_labels, y_pred,filename2)
    plot_model(model, to_file=filename3, show_shapes=True)
    #model.save(out_folder+'\\weights.hdf5')
    
    return auc, accuracy, precision, recall, f1, average_precision


# In[43]:


out_folder = "D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Result\\LSTM\\smote_lstm_256"
LSTM(input_data,256,5,16,out_folder)


# # Bidirectional LSTM

# In[10]:


def BI_LSTM(data, batch_size,epochs_number,emb_output, out_folder):
    
    
    max_features = int(max(np.max(data.train_data), np.max(data.eval_data)))
    min_features= int(min(np.min(data.train_data), np.min(data.eval_data)))
    
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(data.max_input_length,)))
    model.add(tf.keras.layers.Embedding(input_dim=max_features + 1,output_dim=emb_output,mask_zero=True))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, recurrent_dropout=0.1, dropout=0.1)))
    #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, recurrent_dropout=0.1, dropout=0.1)))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1,mode='auto')
    
    history= model.fit(data.train_data, data.train_labels, validation_split=0.2, epochs=epochs_number,batch_size=batch_size, verbose=1, shuffle=True).history
    
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    prob = model.predict(data.eval_data)
    y_pred = get_predicted_y(prob, 0.5)
    try:
        auc, accuracy, precision, recall, f1, average_precision, fpr, tpr = get_all_metrics(model, data.eval_data, data.eval_labels, y_pred)
    except Exception as e:
        print(e)
    
    filename1= out_folder+'\\roc_curve.png' 
    filename2= out_folder+'\\precision_recall_curve.png' 
    filename3= out_folder+'\\model.png' 
              
    save_roc_curve(data.eval_labels, y_pred, filename1)
    save_precision_recall_curve(data.eval_labels, y_pred,filename2)
    plot_model(model, to_file=filename3, show_shapes=True)
    #model.save(out_folder+'\\weights.hdf5')
    
    return auc, accuracy, precision, recall, f1, average_precision


# In[11]:


out_folder = "D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Result\\BI_LSTM\\smote_bi_lstm_256"
BI_LSTM(input_data,256,5,16,out_folder)


# # GRU

# In[12]:


def GRU(data, batch_size,epochs_number,emb_output, out_folder):
    
    
    max_features = int(max(np.max(data.train_data), np.max(data.eval_data)))
    min_features= int(min(np.min(data.train_data), np.min(data.eval_data)))
    
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(data.max_input_length,)))
    model.add(tf.keras.layers.Embedding(input_dim=max_features + 1,output_dim=emb_output,mask_zero=True))

    #model.add(tf.keras.layers.GRU(64, return_sequences=True, recurrent_dropout=0.1, dropout=0.1))
    model.add(tf.keras.layers.GRU(32, return_sequences=True, recurrent_dropout=0.1, dropout=0.1))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1,mode='auto')
    
    history= model.fit(data.train_data, data.train_labels, validation_split=0.2, epochs=epochs_number,batch_size=batch_size, verbose=1, shuffle=True).history
    
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    prob = model.predict(data.eval_data)
    y_pred = get_predicted_y(prob, 0.5)
    try:
        auc, accuracy, precision, recall, f1, average_precision, fpr, tpr = get_all_metrics(model, data.eval_data, data.eval_labels, y_pred)
    except Exception as e:
        print(e)
    
    filename1= out_folder+'\\roc_curve.png' 
    filename2= out_folder+'\\precision_recall_curve.png' 
    filename3= out_folder+'\\model.png' 
    
    save_roc_curve(data.eval_labels, y_pred, filename1)
    save_precision_recall_curve(data.eval_labels, y_pred,filename2)
    plot_model(model, to_file=filename3, show_shapes=True)
    #model.save(out_folder+'\\weights.hdf5')
    
    return auc, accuracy, precision, recall, f1, average_precision


# In[13]:


out_folder = "D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Result\\GRU\\smote_gru_256"
GRU(input_data,256,5,16,out_folder)


# # Bidirectional GRU

# In[14]:


def BI_GRU(data, batch_size,epochs_number,emb_output, out_folder):
    
    
    max_features = int(max(np.max(data.train_data), np.max(data.eval_data)))
    min_features= int(min(np.min(data.train_data), np.min(data.eval_data)))
    
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(data.max_input_length,)))
    model.add(tf.keras.layers.Embedding(input_dim=max_features + 1,output_dim=emb_output,mask_zero=True))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, recurrent_dropout=0.1, dropout=0.1)))
    #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True, recurrent_dropout=0.1, dropout=0.1)))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1,mode='auto')
    
    history= model.fit(data.train_data, data.train_labels, validation_split=0.2, epochs=epochs_number,batch_size=batch_size, verbose=1, shuffle=True).history
    
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    prob = model.predict(data.eval_data)
    y_pred = get_predicted_y(prob, 0.5)
    try:
        auc, accuracy, precision, recall, f1, average_precision, fpr, tpr = get_all_metrics(model, data.eval_data, data.eval_labels, y_pred)
    except Exception as e:
        print(e)
    
    filename1= out_folder+'\\roc_curve.png' 
    filename2= out_folder+'\\precision_recall_curve.png' 
    filename3= out_folder+'\\model.png' 
              
    save_roc_curve(data.eval_labels, y_pred, filename1)
    save_precision_recall_curve(data.eval_labels, y_pred,filename2)
    plot_model(model, to_file=filename3, show_shapes=True)
    #model.save(out_folder+'\\weights.hdf5')
    
    return auc, accuracy, precision, recall, f1, average_precision


# In[15]:


out_folder = "D:\\Master\\Thesis\\CodeSmellsDetector\\DL_Approach\\Result\\BI_GRU\\smote_bi_gru_256"
BI_GRU(input_data,256,5,16,out_folder)

