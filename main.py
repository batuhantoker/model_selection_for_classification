from itertools import combinations
from os import system, listdir
import sys
import numpy as np
import requests
import urllib
import pandas as pd
import tkinter as tk
import tarfile, glob, scipy, time
from tkinter import PhotoImage, filedialog
import signal_features
import scienceplots
from PIL import Image, ImageTk
from signal_features import *
#from model_selection import *
import urllib.request
import os
from utils import *
import matplotlib.pyplot as plt
from sklearn import datasets
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use(['science','no-latex','grid'])
# urllib.request.urlretrieve('https://www.dropbox.com/s/kxrqhqhcz367v77/nina.tar.gz?dl=1', "nina.tar.gz")
dir = os. getcwd()
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
    print((mat_ks))
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

import random

def model_optimizer(feature_set, labels, find_best, models, random_seed):
    random.seed(random_seed)
    if find_best == "feature":
        print('You provided following models:')
        for (i, item) in enumerate(models, start=0): print(i, item[1]);
        index = int(input("Enter the index of the model from the given models list: "))
        print(f'Looking for best features for {models[index][1]}')
        find_best_feature(feature_set, labels, models[index])
    elif find_best == "model":
        print('Looking for best feature set in following models:')
        for (i, item) in enumerate(models, start=0): print(i, item[1]);
        find_best_model(feature_set, labels, models)
    elif find_best == "feature-model":
        print('Looking for best feature set and model pair in following models:')
        for (i, item) in enumerate(models, start=0): print(i, item[1]);
        find_all_best(feature_set, labels, models)
    else:
        raise ValueError("Invalid value for 'find_best' parameter. Must be one of 'feature', 'model', or 'feature-model'.")

def find_best_feature(feature_set, labels, model):
    # code for finding the best feature for the given model
    pass

def find_best_model(feature_set, labels, models):
    # code for finding the best model among the models given in the models list
    pass

def find_all_best(feature_set, labels, models):
    # code for finding the best combination of feature and model among the models given in the models list
    pass

def TOBECHECK(nono):
    exercise = 3
    subject = 2
    emg, stimulus, glove = get_data(subject, exercise)
    epoch = 50
    RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM = signal_features.features(emg, epoch)
    feature_set = {'rms': RMS, 'mav': MAV, 'iav': IAV, 'var': VAR,
                   'wl': WL, 'mf': MF, 'pf': PF, 'mp': MP,
                   'tp': TP, 'sm': SM}
    feature_set = pd.DataFrame.from_dict(feature_set.items()).reset_index()

    feature_set_xarray = np.squeeze([[feature_set.loc[i][1]] for i in range(feature_set.shape[0])])
    # feature_set_xarray = feature_set_xarray.reshape()
    labels = signal_features.labels(stimulus, epoch)
    labels = np.nan_to_num(labels)
    feature_set_xarray_n = feature_set_xarray.reshape(feature_set_xarray.shape[1],
                                                      feature_set_xarray.shape[0] * feature_set_xarray.shape[2])
    feature_set_xarray_n = np.nan_to_num(feature_set_xarray_n)

    models = [('LogReg', LogisticRegression(solver='lbfgs', max_iter=1000)),
              ('SVM', SVC()),
              ('DecTree', DecisionTreeClassifier()),
              ('KNN', KNeighborsClassifier(n_neighbors=15)),
              ('LinDisc', LinearDiscriminantAnalysis()),
              # ('GaussianNB', GaussianNB()),
              ('MLPC', MLPClassifier(activation='relu', solver='adam', max_iter=500)),
              # ('RFC',RandomForestClassifier()),
              # ('ABC', AdaBoostClassifier())
              ]
    # model_optimizer(feature_set_xarray_n,labels,'feature',models,42)
    n_folds = 3
    # signal_features.classifier(feature_set_xarray_n, labels, n_folds)

    min_accuracy = 0.9
    # Initialize variables to store the best model and its performance
    best_model = []
    best_accuracy = 0
    # Feature and model selection selection
    for i in range(1, feature_set.shape[0]):
        for features in combinations(range(10), i):
            print(features)
            feature_names = [feature_set[0][k] for k in features]
            feature_idx = features
            cur_X = feature_set_xarray[features, :, :]
            cur_X = cur_X[~np.isnan(cur_X)].reshape(cur_X.shape[1], cur_X.shape[0] * cur_X.shape[2])

            # print(cur_X.shape)
            # X_train, X_test, y_train, y_test = train_test_split(cur_X, labels, test_size=0.2)
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
                    # f1 = f1_score(y_test, y_pred)
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

    # cur_X = feature_set_xarray[0:1, :, :]
    # print(cur_X.shape)
    # selected_X = cur_X[~np.isnan(cur_X)].reshape(cur_X.shape[1], cur_X.shape[0] * cur_X.shape[2])
    # signal_features.classifier(selected_X, labels, n_folds)

def data_from_local():
    global data
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv"),("MAT Files","*.mat"),("TXT Files","*.txt")])
    file_ext = filepath.split(".")[-1]
    if file_ext == "csv":
        data = pd.read_csv(filepath)
    elif file_ext == "mat":
        data = scipy.io.loadmat(filepath)
    elif file_ext == "txt":
        data = pd.read_csv(filepath, delimiter = '\t')
    display_data()

def data_from_web():
    def import_online():
        global data
        print(dataset_var.get())
        dataset_name = dataset_var.get()#dataset_names[int(dataset_var.get())]
        print(dataset_name)
        data = getattr(datasets, dataset_name)()
        display_data2(import_screen)
        import_screen.destroy()


    import_screen = tk.Tk()
    import_screen.title("Import Data")
    dataset_var = tk.StringVar(value="Select from listed datasets.")
    dataset_names = ['load_iris', 'load_boston', 'load_diabetes', 'fetch_california_housing', 'fetch_olivetti_faces']
    dataset_select_label = tk.Label(import_screen, text="Select from listed datasets.")
    dataset_select_label.pack()
    tk.OptionMenu(import_screen, dataset_var, *dataset_names).pack()
    select_button = tk.Button(import_screen, text="Select", command=import_online).pack()

    label = tk.Label(import_screen)
    label.pack()
    import_screen.mainloop()


def import_data2():
    global data
    dataset_name = dataset_var.get()
    data = getattr(datasets,dataset)()
    display_data2()

def display_data2(import_screen):
    global data, target, features
    data = pd.DataFrame(data=np.c_[data['data'], data['target']],
                         columns=data['feature_names'] + ['target'])
    data = data.dropna()

    def store_selections(target_var, feature_vars):
        global data, target, features

        target = (data[target_var.get()])#data.columns[str( )]
        features = pd.DataFrame()
        for i in feature_vars:
            print({data.columns[i]: (data[data.columns[i]])})
            features[data.columns[i]] = np.asarray(data[data.columns[i]])# features.append(features_temp, ignore_index=True) #features[] = data[data.columns[i]]
        print(features)
        target_text = tk.Label(display_window, text="Data is selected. You can return this page to change selected features.")
        target_text.pack()
        framework_selection()
    display_window = tk.Toplevel()
    display_window.title("Data Preview")
    target_var = tk.StringVar(value='target')
    feature_vars = tk.StringVar()
    target_label = tk.Label(display_window, text="Select Target Column")
    target_label.pack()
    target_select = tk.OptionMenu(display_window, target_var, *list(data.columns))
    target_select.pack()
    feature_label = tk.Label(display_window, text="Select Feature Columns")
    feature_label.pack()
    feature_select = tk.Listbox(display_window, selectmode='multiple', listvariable=feature_vars)
    for col in data.columns:
        feature_select.insert(tk.END, col)
    feature_select.pack()
    submit_button = tk.Button(display_window, text="Submit", command=lambda: store_selections(target_var,feature_select.curselection()))
    submit_button.pack()
    data_label = tk.Label(display_window, text=data.head().to_string())
    data_label.pack()



def intro_page():
    root.title("Import Data")
    image0 = ImageTk.PhotoImage(Image.open("img/local.png").resize((200, 200)))
    local_button = tk.Button(root,image=image0, command=data_from_local)
    local_label = tk.Label(root, text="Select from Local")
    local_label.pack()
    local_button.pack()

    image2 = ImageTk.PhotoImage(Image.open("img/web.png").resize((200, 200)))
    web_button = tk.Button(root,image=image2, command=data_from_web)
    web_label = tk.Label(root, text="Select from Web")
    web_label.pack()
    web_button.pack()

    root.mainloop()

def framework_selection():
    global data, target, features,root
    root.destroy()
    root = tk.Tk()




    root.geometry("600x600")
    root.title("Selection Window")

    image_sp = ImageTk.PhotoImage(Image.open("img/spark.png").resize((200, 200)))
    pyspark_button = tk.Button(root,image=image_sp, command=pySpark_func)
    pyspark_label = tk.Label(root, text="pySpark")


    image_sk = ImageTk.PhotoImage(Image.open("img/sklearn.png").resize((200, 200)))
    sklearn_button = tk.Button(root,image=image_sk, command=sklearn_func)
    sklearn_label = tk.Label(root, text="SK-Learn")




    # Pack radio buttons and next button
    pyspark_label.pack()
    pyspark_button.pack()
    sklearn_label.pack()
    sklearn_button.pack()



    root.mainloop()





def sklearn_func():

    global models_param,root
    models_param = {}
    #root.destroy()
    new_root = tk.Tk()
    new_root.title("Parameter Configuration")
    model_var = tk.StringVar()
    model_var.trace("w", lambda *args: get_parameters(model_var.get()))

    def get_parameters(models_name):
        global models_param, models_list, target, features
        models_name = models_name.split(',')

        #parameter_frame = tk.Tk()
        # for model_name in models_name:
            # if model_name == "Linear Regression":
            #     models_list.append(LinearRegression())
            # elif model_name == "Decision Tree":
            #     models_list.append(DecisionTreeClassifier())
            # elif model_name == "Support Vector Machine":
            #     models_list.append(svm.SVC())
            # else:
            #     return
            # parameters = model.get_params()
            # print(len(parameters))
            # for key, value in parameters.items():
            #     print(key,value)
            #     label = tk.Label(parameter_frame, text=f"{key} : {value}")
            #     label.pack()
            #     print(type(value))
            #     entry = tk.Entry(parameter_frame, textvariable=tk.StringVar(value))
            #     if type(value) == str:
            #         entry.insert(tk.END, value)
            #     else:
            #         entry.insert(tk.END, value)
            #     entry.pack()
            #     models_param[model_name] = {key: entry}

    def on_select(var_list,model_list):
        global models_list
        selection = [model_list[i] for i in range(len(var_list)) if var_list[i].get()]
        models_list = []
        for model_name in selection:
            if model_name == "Decision Tree":
                models_list.append(('Decision Tree',DecisionTreeClassifier()))
            elif model_name == "Support Vector Machine":
                models_list.append(('SVM',svm.SVC()))
            elif model_name == "Logistic Regression":
                models_list.append(('LogReg', LogisticRegression()))
            elif model_name == "KNeighborsClassifier":
                models_list.append(('KNN', KNeighborsClassifier()))
            elif model_name == "GaussianNB":
                models_list.append(('GaussianNB', GaussianNB()))
            elif model_name == "MLPC":
                models_list.append(('MLPC', KNeighborsClassifier()))
            elif model_name == "RandomForestClassifier":
                models_list.append(('RandomForestClassifier', RandomForestClassifier()))
            elif model_name == "AdaBoostClassifier":
                models_list.append(('AdaBoostClassifier', AdaBoostClassifier()))
            elif model_name == "LinearDiscriminantAnalysis":
                models_list.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
            elif model_name == "KNeighborsClassifier":
                models_list.append(('KNN', KNeighborsClassifier()))
        print(models_list)

    var_list=[]
    set_button = tk.Button(new_root, text="Set", command=on_train_sklearn)
    set_button.pack()
    model_list = ["Logistic Regression", "Support Vector Machine",  "Decision Tree", "KNeighborsClassifier","GaussianNB" ,"MLPC","RandomForestClassifier","AdaBoostClassifier","LinearDiscriminantAnalysis"]
    for element in model_list:
        var = tk.IntVar()
        var_list.append(var)
        cb = tk.Checkbutton(new_root, text=element, variable=var, command=lambda: on_select(var_list,model_list))
        cb.pack()

    new_root.mainloop()


def on_train_sklearn():
    global models_param, models_list, target, features
    X = features
    y = target
    print(X.shape,y.shape)
    def train_models():
        global X_train, X_test, y_train, y_test, trained_models
        trained_models=[]
        for model in models_list:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio_var.get()))
            start_time = time.time()
            print(model[1])
            trained_models.append(model[1].fit(X_train, y_train))
            end_time = time.time()

            execution_time = round(end_time - start_time, 3)
            train_label = tk.Label(on_train_window, text=f"{model[0]} is successfully trained in {execution_time} seconds.")
            train_label.pack()
            print(f"{model[0]} is successfully trained in {execution_time} seconds.")

    def test_models():
        global X_train, X_test, y_train, y_test, trained_models
        for index,model in enumerate(trained_models):
            accuracy = model.score(X_test, y_test)
            print(f"{models_list[index][0]} has an accuracy of {accuracy}.")
            test_label = tk.Label(on_train_window,
                                   text=f"{models_list[index][0]} has an accuracy of {accuracy}.")
            test_label.pack()


    on_train_window = tk.Tk()
    on_train_window.geometry("600x600")

    # Create checkboxes for each model
    #print(models_list)


    # Create input for train/test split ratio
    split_ratio_var = tk.DoubleVar(value=0.8)
    split_ratio_entry = tk.Entry(on_train_window, textvariable=split_ratio_var)
    split_ratio_entry.insert(tk.END, 0.8)
    split_ratio_entry.pack()

    # Create train button
    train_button = tk.Button(on_train_window, text="Train", command=train_models)
    train_button.pack()

    # Create test button
    test_button = tk.Button(on_train_window, text="Test", command=test_models)
    test_button.pack()

    on_train_window.mainloop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #intro_page()
    global data, target, features, root

    iris = datasets.load_iris()
    features = iris.data[:, :2]  # we only take the first two features.
    target = iris.target
    #root = tk.Tk()
    #intro_page()
    #framework_selection()
    pySpark_func()
    #sklearn_func()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
