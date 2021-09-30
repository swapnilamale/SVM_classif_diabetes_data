# -*- coding: utf-8 -*-

# SVM classification
# dataset: diabetes prediction (binary classification)

# libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, classification_report

# read the file
path="F:/aegis/4 ml/dataset/supervised/classification/diabetes/diab.csv"

diab = pd.read_csv(path)

diab.shape
diab.head()
diab.tail()

# since 'class_val' is repeating, remove this from the dataset
diab.drop(columns='class_val',inplace=True)
diab.columns

# check the y-distribution
diab["class"].value_counts()

# include EDA code here

# standardize the data
def stdData(data,y,std):
    
    D = data.copy()
    
    if std == "ss":
        tr = preprocessing.StandardScaler()
    elif std == "minmax":
        tr = preprocessing.MinMaxScaler()
    else:
        return("Invalid Type specified")
    
    D.iloc[:,:] = tr.fit_transform(D.iloc[:,:])
    
    # restore the actual y-value
    D[y] = data[y]
    
    return(D)


# standard scaler data
diab_ss = stdData(diab,"class","ss")

# minmax scaler data
diab_mm = stdData(diab,"class","minmax")  

diab_ss.head()
diab_mm.head()

# split the stdscaler data into train and test
trainx1,testx1,trainy1,testy1 = train_test_split(diab_ss.drop("class",1),
                                                 diab_ss["class"],test_size=0.25)
print(trainx1.shape,trainy1.shape)
print(testx1.shape,testy1.shape)

# split the minmaxscaler data into train and test
trainx2,testx2,trainy2,testy2 = train_test_split(diab_mm.drop("class",1),
                                                 diab_mm["class"],test_size=0.25)

print(trainx2.shape,trainy2.shape)
print(testx2.shape,testy2.shape)

# SVM specific parameters

# list of values for C and gamma
lim=10
lov_c = np.logspace(-5,4,lim)
lov_g = np.random.random(lim)

'''
kernels in SVM
linear -> C
sigmoid -> C,gamma
poly -> C,gamma
rbf(radial basis function) -> C,gamma
'''

# build the parameters
params = [{ 'kernel':['linear'], 'C':lov_c, 
            'kernel':['sigmoid'],'C':lov_c,'gamma':lov_g,
            'kernel':['poly'],'C':lov_c,'gamma':lov_g,
            'kernel':['rbf'],'C':lov_c,'gamma':lov_g}]


# perform Grid Search
model = svm.SVC()

# grid CV on stdscaler data
grid = GridSearchCV(model,param_grid=params,
                    scoring="accuracy",cv=3,
                    n_jobs=-1).fit(trainx1,trainy1)

# best parameters
bp = grid.best_params_
bp1 = bp.copy()
bp1

# build the model with the best parameters
m1 = svm.SVC(kernel=bp['kernel'], C=bp['C'], gamma=bp['gamma']).fit(trainx1,trainy1)

# predictions
p1 = m1.predict(testx1)

def cm(actual,pred):
    # model accuracy
    print("Model Accuracy = {}".format(accuracy_score(actual,pred)))
    print("\n")
    
    # confusion matrix
    df = pd.DataFrame({'actual':actual,'pred':pred})
    print(pd.crosstab(df.actual,df.pred,margins=True))
    print("\n")
    
    # classification report
    print(classification_report(actual,pred))
    
    return(1)

cm(testy1,p1)


# grid CV on stdscaler data
grid = GridSearchCV(model,param_grid=params,
                    scoring="accuracy",cv=3,
                    n_jobs=-1).fit(trainx2,trainy2)

bp2 = grid.best_params_

# build the model with the best parameters
m2 = svm.SVC(kernel=bp2['kernel'], C=bp2['C'], gamma=bp2['gamma']).fit(trainx2,trainy2)

# predictions
p2 = m2.predict(testx2)

cm(testy2,p2)    

    
    

