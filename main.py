from itertools import combinations
from os import system, listdir
import sys
import numpy as np
import requests
import urllib
import pandas as pd
import tarfile, glob, scipy
import signal_features
import scienceplots
from signal_features import *
#from model_selection import *
import urllib.request
import os
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use(['science','no-latex','grid'])
# urllib.request.urlretrieve('https://www.dropbox.com/s/kxrqhqhcz367v77/nina.tar.gz?dl=1', "nina.tar.gz")

src_path = 'nina.tar.gz'
dst_path = 'C:\\Users\\batua\\PycharmProjects\\model_selection_for_classification\\data'

# if src_path.endswith('tar.gz'):
#     tar = tarfile.open(src_path, 'r:gz')
#     tar.extractall(dst_path)
#     tar.close()
def get_data(subject,exercise):
    directories=glob.glob("data/**/**/*.mat")
    ends_with = f'S{subject}_E{exercise}_A1.mat'
    new_directories = [dir for dir in directories if dir.endswith(ends_with)]
    mat = scipy.io.loadmat(new_directories[0])
    mat_ks = [k for k in mat.keys()]
    # print((mat_ks))
    df = pd.DataFrame.from_dict(mat.items())
    # print(df)
    # print(df.loc[3][1].shape)
    emg = df.loc[3][1]
    # Filtering
    fs = 200
    lowcut = 20
    highcut = 80
    cutoff = 60
    print(df,emg.shape)
    emg = signal_features.data_preprocess(emg,fs,lowcut,highcut,cutoff)
    pca_x = PCA(n_components=12)

    emg_new = pca_x.fit_transform(emg)
    print(emg_new.shape)
    exp_var_pca = pca_x.explained_variance_ratio_
    print(np.sum(exp_var_pca))
    emg=emg_new
    stimulus = df.loc[10][1]
    # stimulus_p = np.nan_to_num(stimulus)
    # print(np.where(stimulus==0))
    # stimulus=np.delete(stimulus_p,np.where(stimulus_p==0))
    #
    # emg_new=np.delete(emg,np.where(stimulus_p==0), axis=0)
    # emg=emg_new

    glove = df.loc[6][1]
    return emg,stimulus,glove






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    exercise = 3
    subject = 2
    emg, label, glove = get_data(subject,exercise)
    epoch = 50
    RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM = signal_features.features(emg, epoch)
    feature_set = {'rms': RMS, 'mav': MAV, 'iav': IAV, 'var': VAR,
                'wl': WL, 'mf': MF, 'pf': PF, 'mp': MP,
                'tp': TP, 'sm': SM}
    feature_set = pd.DataFrame.from_dict(feature_set.items()).reset_index()
    feature_set_xarray = np.squeeze([[feature_set.loc[i][1]] for i in range(feature_set.shape[0])])
    #feature_set_xarray = feature_set_xarray.reshape()
    labels = signal_features.labels(stimulus, epoch)
    labels = np.nan_to_num(labels)
    feature_set_xarray_n =feature_set_xarray.reshape(feature_set_xarray.shape[1],feature_set_xarray.shape[0]*feature_set_xarray.shape[2])
    feature_set_xarray_n =np.nan_to_num(feature_set_xarray_n)

    #print(feature_set[0][3])
    models = [#('LogReg', LogisticRegression(solver='lbfgs', max_iter=1000)),
              #('SVM', SVC()),
              # ('DecTree', DecisionTreeClassifier()),
              # ('KNN', KNeighborsClassifier(n_neighbors=15)),
              ('LinDisc', LinearDiscriminantAnalysis()),
              # ('GaussianNB', GaussianNB()),
              ('MLPC', MLPClassifier(activation='relu', solver='adam', max_iter=500)),
              # ('RFC',RandomForestClassifier()),
              # ('ABC', AdaBoostClassifier())
              ]
    n_folds=2
    min_accuracy = 0.9
    # Initialize variables to store the best model and its performance
    best_model = []
    best_accuracy = 0
    # Feature and model selection selection
    for i in range(1, feature_set.shape[0]):
        for features in combinations(range(10), i):
            feature_names = [feature_set[0][k] for k in features]
            feature_idx = features
            cur_X = feature_set_xarray[features,:,: ]
            cur_X = cur_X[~np.isnan(cur_X)].reshape(cur_X.shape[1],cur_X.shape[0]*cur_X.shape[2])

            #print(cur_X.shape)
            #X_train, X_test, y_train, y_test = train_test_split(cur_X, labels, test_size=0.2)
            for train_idx, test_idx in KFold(n_folds).split(cur_X):
                X_train, X_test = cur_X[train_idx], cur_X[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]
                # Iterate over the models
                for model_name, model in models:
                    # Fit the model on the training data
                    model.fit(X_train, y_train)

                    # Evaluate the model on the test data
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    #f1 = f1_score(y_test, y_pred)
                    print(f'{model_name}, with {len(feature_names)} feature(s), Accuracy: {accuracy}, not good enouh')

                    # Update the best model and its performance if necessary
                    if accuracy >= min_accuracy or accuracy >= best_accuracy:
                        best_model.append([model_name, feature_names, accuracy])
                        print('New good performance!')
                        # Print the performance of the model
                        print('Model: {}'.format(model_name))
                        print('Features: {}'.format(feature_names))
                        print('Accuracy: {}'.format(accuracy))
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model_name = model_name
                            best_features = feature_names
                            print(f'New best performance! :{model_name}')

    print(f'Code is completed. Best performance features and models are:')
    print(f'Best model : {best_model_name}, with {best_accuracy} accuracy with {best_features} features)')
    print(best_model)

    #signal_features.classifier(feature_set_xarray_n,labels,2)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
