import numpy as np
from sklearn import preprocessing,cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score
from pandas.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt


#Loading the data set from uci
url='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
names=['id','clump_thickness','uniform_cell_size','uniform_cell_shape',
      'marginal_adhesion','signle_epithelial_size','bare_nuclei',
      'bland Chromatin','normal_nucleoli','Mitoses','class']
Data=pd.read_csv(url,names=names)


#preprocess the data
Data.replace('?',-99999,inplace=True)
print(Data.axes)  #print data axes
Data.drop(['id'],1,inplace=True)
print(Data.shape)  #print data shape


#data visualization 
print(Data.loc[0])
print(Data.describe())



#plot histogram for each variable
Data.hist(figsize=(10,10))
plt.show()


#scatter plot matrix
scatter_matrix(Data,figsize=(18,18))
plt.show()



#create x and y data
x=np.array(Data.drop(['class'],1))
y=np.array(Data['class'])

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)


#specifiy testing option 
seed=8
scoring='accuracy'


#define the model to train
models=[]
models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM',SVC()))

#evaluate each model by running for loop on all the data with the 2 models
results=[]
names=[]

for name,model in models :
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model,x_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(names)
    
    msg="%s %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)


# make prediction on the test set 
for name,model in models:
    model.fit(x_train,y_train)
    predictions=model.predict(x_test)
    print(name)
    print(accuracy_score(y_test,predictions))
    print(classification_report(y_test,predictions))

