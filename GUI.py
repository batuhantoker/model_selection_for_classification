import tkinter as tk
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

models = [('LogReg', LogisticRegression(solver='lbfgs', max_iter=1000)),
          ('SVM', SVC()),
          ('DecTree', DecisionTreeClassifier()),
          ('KNN', KNeighborsClassifier(n_neighbors=15)),
          ('LinDisc', LinearDiscriminantAnalysis()),
          ('MLPC', MLPClassifier(activation='relu', solver='adam', max_iter=500))
         ]
features = ['RMS', 'MAV', 'IAV', 'VAR', 'WL', 'MF', 'PF', 'MP', 'TP', 'SM']

def on_select():
    subject = subject_var.get()
    exercise = exercise_var.get()
    print("Selected Subject: ", subject)
    print("Selected Exercise: ", exercise)
    selected_models = [model[0] for model in models if model[0] in [var.get() for var in model_vars]]
    selected_features = [var.get() for var in feature_vars if var.get()]
    print("Selected Models: ", selected_models)
    print("Selected Features: ", selected_features)
    message = tk.Label(root, text="Dataset Selected", fg="green")
    message.grid(row=4, column=3, columnspan=2)

def display_text():
    message2 = tk.Label(root, text="Dataset Selected", fg="green")
    message2.grid(row=0, column=7, columnspan=2)

root = tk.Tk()
root.title("Subject and Exercise Selector")

subject_var = tk.IntVar(value=1)
exercise_var = tk.IntVar(value=1)
find_best_var = tk.StringVar(value="features")

subject_label = tk.Label(root, text="Select Subject:")
subject_label.grid(row=0, column=0)

subject_menu = tk.OptionMenu(root, subject_var, *range(1, 11))
subject_menu.grid(row=0, column=1)

exercise_label = tk.Label(root, text="Select Exercise:")
exercise_label.grid(row=1, column=0)

exercise_menu = tk.OptionMenu(root, exercise_var, *range(1, 4))
exercise_menu.grid(row=1, column=1)

select_button = tk.Button(root, text="Select", command=on_select)
select_button.grid(row=2, column=0, columnspan=2)

model_frame = tk.LabelFrame(root, text="Select Models")
model_frame.grid(row=0, column=2, rowspan=3)

model_vars = [tk.StringVar() for _ in range(len(models))]
for var in model_vars:
    var.set(False)
for i, (name, model) in enumerate(models):
    tk.Checkbutton(model_frame, text=name, variable=model_vars[i]).pack()

feature_frame = tk.LabelFrame(root, text="Select Features")
feature_frame.grid(row=4, column=2, columnspan=2)

feature_vars = [tk.StringVar() for _ in range(len(features))]
for var in feature_vars:
    var.set(False)
for i, feature in enumerate(features):
    tk.Checkbutton(feature_frame, text=feature, variable=feature_vars[i]).pack()

find_best_frame = tk.LabelFrame(root, text="Find Best")
find_best_frame.grid(row=0, column=4, columnspan=2)

tk.Radiobutton(find_best_frame, text="Features", variable=find_best_var, value="features").pack()
tk.Radiobutton(find_best_frame, text="Model", variable=find_best_var, value="model").pack()
tk.Radiobutton(find_best_frame, text="Features and Model", variable=find_best_var, value="features and model").pack()


root.mainloop()
