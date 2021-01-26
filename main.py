import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV

         
le = preprocessing.LabelEncoder()

#read train data
df = pandas.read_csv('1601355459_1343217_train.csv')

features = ['F1', 'F2', 'F3', 'F4','F5', 'F6', 'F7', 'F8','F9']

X_train = df[features]



# Fit the encoder to the pandas column
le.fit(df['F10'])    
X_F10=le.fit_transform(df['F10']) #convert string label to number


# Fit the encoder to the pandas column
le.fit(df['F11']) 
X_F11=le.fit_transform(df['F11']) #convert string label to number


X_train['F10']=X_F10   
X_train['F11']=X_F11       #train data


y_train = df['credit']     #train target
n=len(y_train)

#best feature selection
X_new = SelectKBest(chi2, k=7).fit_transform(X_train, y_train)


#==================  decision tree  =========================
#cross validation using grid search 
parameters = {'max_depth':range(3,20)}
clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
clf.fit(X=X_new, y=y_train)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_) 

# predict
y_pred = tree_model.predict(X_new)

#confusion matrix and accuracy 
#accuracy

acc=accuracy_score(y_train, y_pred)
print('accuracy(decision tree): ',acc*100)

#confusion matrix plot 


title=("Confusion matrix of decision tree")

disp = plot_confusion_matrix(tree_model, X_new,y_train)
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)



#============================================================

#==================  knn  =========================
#cross validation using grid search 

k_range = list(range(1,10))
weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors = k_range, weights = weight_options)
#print (param_grid)
knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid, cv = 3, scoring = 'accuracy')

grid.fit(X_new,y_train)

knn_model = grid.best_estimator_
print (grid.best_score_, grid.best_params_) 

# predict
y_pred = grid.predict(X_new)

#confusion matrix and accuracy 
#accuracy

acc=accuracy_score(y_train, y_pred)
print('accuracy(knn): ',acc*100)

#confusion matrix plot 


title=("Confusion matrix of knn")

disp = plot_confusion_matrix(grid, X_new,y_train)
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)


#============================================================




# =====================predict test data=======================

#read test data
df = pandas.read_csv('1601355459_1391656_test.csv')
features = ['F1', 'F2', 'F3', 'F4','F5', 'F6', 'F7']
X_test = df[features]




# predict
Z_dt = tree_model.predict(X_test)

#=============================================================
with open('Format File.txt', 'w') as log:
      for x in Z_dt:
          log.write(str(x)+'\n')                
