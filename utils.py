import pyspark.ml.classification
from pyspark import SparkContext
from pyspark.sql import SQLContext
import tkinter as tk
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
def pySpark_func():
    global data, target, features
    global models_param
    new_root = tk.Tk()
    # sc = SparkContext().getOrCreate()
    # sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName("churn").getOrCreate()
    #dataset = spark.read.csv("../datasets/customer_churn.csv", inferSchema=True, header=True)
    data.printSchema()
    def on_select(var_list,model_list):
        global models_list
        selection = [model_list[i] for i in range(len(var_list)) if var_list[i].get()]
        print(selection)
        models_list = []
        for model_name in selection:
            if model_name == "Decision Tree":
                models_list.append(('Decision Tree',pyspark.ml.classification.DecisionTreeClassifier()))
            elif model_name == "Support Vector Machine":
                models_list.append(('SVM', pyspark.ml.classification.LinearSVC()))
            elif model_name == "Logistic Regression":
                models_list.append(('LogReg', pyspark.ml.classification.LogisticRegression()))
            elif model_name == "GaussianNB":
                models_list.append(('GaussianNB', pyspark.ml.classification.NaiveBayes()))
            elif model_name == "MLPC":
                models_list.append(('MLPC', pyspark.ml.classification.MultilayerPerceptronClassifier()))
            elif model_name == "RandomForestClassifier":
                models_list.append(('RandomForestClassifier', pyspark.ml.classification.RandomForestClassifier()))
        print(models_list)

    var_list=[]
    set_button = tk.Button(new_root, text="Set", command=on_train_pyspark)
    set_button.pack()
    model_list = ["Logistic Regression", "Support Vector Machine",  "Decision Tree","GaussianNB" ,"MLPC","RandomForestClassifier"]
    for element in model_list:
        var = tk.IntVar()
        var_list.append(var)
        cb = tk.Checkbutton(new_root, text=element, variable=var, command=lambda: on_select(var_list,model_list))
        cb.pack()

    new_root.mainloop()

def on_train_pyspark():
    pass