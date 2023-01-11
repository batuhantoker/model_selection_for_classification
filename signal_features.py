from sklearn import *
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
import pickle
from sklearn.preprocessing import MinMaxScaler
import pickle
from itertools import chain
import math
import numpy as np
from sklearn.decomposition import PCA
from scipy import signal
from scipy.signal import butter, sosfilt, sosfreqz, lfilter, freqz
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from numpy.fft import fft, ifft
from scipy.optimize import curve_fit
#from tfestimate import *
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from numpy.lib.stride_tricks import sliding_window_view
#from sklearn.metrics import plot_confusion_matrix
import numpy as np
from scipy import  signal
import math
from sklearn import preprocessing
import warnings

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def zero_lag_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data, padlen=15)
    return y

def data_preprocess(emg_data,fs,lowcut,highcut,cutoff):
    print('Data filtering...')
    emg_ = zero_lag_filter(emg_data, lowcut, highcut, fs, order=4)
    #emg_ = abs(emg_)
    scaler = preprocessing.MinMaxScaler()
    emg_scaled = scaler.fit_transform(emg_)
    #emg_ = butter_lowpass_filter(emg_rect, cutoff, fs, 4)
    #emg_=samplerate.resample(emg_,0.5)
    return emg_scaled
def features(data, epoch):
    print('Feature extraction...')
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0:number_of_segments * epoch, :], number_of_segments)
    RMS = np.empty([number_of_segments, data.shape[1]])
    MAV = np.empty([number_of_segments, data.shape[1]])
    IAV = np.empty([number_of_segments, data.shape[1]])
    VAR = np.empty([number_of_segments, data.shape[1]])
    WL = np.empty([number_of_segments, data.shape[1]])
    MF = np.empty([number_of_segments, data.shape[1]])
    PF = np.empty([number_of_segments, data.shape[1]])
    MP = np.empty([number_of_segments, data.shape[1]])
    TP = np.empty([number_of_segments, data.shape[1]])
    SM = np.empty([number_of_segments, data.shape[1]])
    # max_ind = np.empty([number_of_segments, 4])
    for i in range(number_of_segments):
        RMS[i, :] = np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
        # max_ind [i,:] = RMS[i,:][np.argpartition(RMS[i,:],5, axis=0)]
        MAV[i, :] = np.mean(np.abs(splitted_data[i]), axis=0)
        IAV[i, :] = np.sum(np.abs(splitted_data[i]), axis=0)
        VAR[i, :] = np.var(splitted_data[i], axis=0)
        WL[i, :] = np.sum(np.diff(splitted_data[i], prepend=0), axis=0)
        freq, power = signal.periodogram(splitted_data[i], axis=0)
        fp = np.empty([len(freq), power.shape[1]])
        for k in range(len(freq)):
            fp[k] = power[k, :] * freq[k]
        MF[i, :] = np.sum(fp, axis=0) / np.sum(power, axis=0)  # Mean frequency
        PF[i, :] = freq[np.argmax(power, axis=0)]  # Peak frequency
        MP[i, :] = np.mean(power, axis=0)  # Mean power
        TP[i, :] = np.sum(power, axis=0)  # Total power
        SM[i, :] = np.sum(fp, axis=0)  # Spectral moment
    return RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM

def labels(data, epoch):
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0:number_of_segments * epoch], number_of_segments)
    class_value = np.empty([number_of_segments])
    for i in range(number_of_segments):
        class_value[i] = np.rint(np.sqrt(np.mean(np.square(splitted_data[i]))))
    return class_value
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)



def data_reshape(data):
    data = np.reshape(data, (len(data),64))  #8, 8
    data = data.astype(np.float64)
    return data

def classifier(features,labels,k_fold):
    Y = labels
    X = features
    number_of_k_fold = k_fold
    random_seed = 42
    outcome = []
    model_names = []
    # Variables for average classification report
    originalclass = []
    classification = []
    models = [('LogReg', LogisticRegression()),
              ('SVM', SVC()),
              ('DecTree', DecisionTreeClassifier()),
              ('KNN', KNeighborsClassifier(n_neighbors=15)),
              ('LinDisc', LinearDiscriminantAnalysis()),
              ('GaussianNB', GaussianNB()),
              ('MLPC', MLPClassifier(activation='relu', solver='adam', max_iter=500)),
              ('RFC',RandomForestClassifier()),
              ('ABC', AdaBoostClassifier())
              ]



    for model_name, model in models:
        k_fold_validation = model_selection.KFold(n_splits=number_of_k_fold, random_state=random_seed, shuffle=True)
        results = model_selection.cross_val_score(model, X, Y, cv=k_fold_validation,
                                                  scoring='accuracy')
        outcome.append(results)
        model_names.append(model_name)
        output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
        print(output_message)
    print(classification)
    fig = plt.figure()
    fig.suptitle('Machine Learning Model Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(outcome)
    plt.ylabel('Accuracy [%]')
    ax.set_xticklabels(model_names)
    fig2 = plt.figure()
    plt.show()
# # load data
# flex_pp = data_reshape(np.loadtxt('flex_pp.txt'))
# ext_pp = data_reshape(np.loadtxt('ext_pp.txt'))
# emg_class = (np.loadtxt('emg_class.txt'))
#
# epoch=100
# emg_class = class_map(emg_class,epoch)
# valid_classes=np.r_[np.array([i for i, v in enumerate(emg_class) if v.is_integer()])]

#
# RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM =activation_map(ext_pp,epoch)
# #MSF = mean_shift_feature(RMS)
# dict_ext = {'rms_ext':RMS, 'mav_ext':MAV, 'iav_ext':IAV, 'var_ext':VAR,'wl_ext':WL, 'mf_ext':MF, 'pf_ext':PF, 'mp_ext':MP, 'tp_ext':TP,'sm_ext':SM} #, 'msf_ext':MSF
# RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM =activation_map(flex_pp,epoch)
# #MSF = mean_shift_feature(RMS)
# dict_flex= {'rms_flex':RMS, 'mav_flex':MAV, 'iav_flex':IAV, 'var_flex':VAR,'wl_flex':WL, 'mf_flex':MF, 'pf_flex':PF, 'mp_flex':MP, 'tp_flex':TP, 'sm_flex':SM} # , 'msf_flex':MSF
# dict_target = {'movement_id': emg_class}
#
#
# z = dict(dict_flex, **dict_ext)
# z2 = dict(z, **dict_target)
# print(z2.keys())
# with open('data11.pkl', 'wb') as handle:
#     pickle.dump(z2, handle, protocol=pickle.HIGHEST_PROTOCOL)

#data7 s1, data8 s2, data9 s3, data10 s4, data11 s1 50-63