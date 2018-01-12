'''
Go to UCI repository. Pick any dataset of your choice from the 
classification task, categorical attributes, and multivariate data.
Develop a Naive Bayes classifier for this dataset. 
Try various train-test splits (at least 5) and report accuracies
of your classifier.    
'''
from itertools import izip
import numpy as np
import sklearn.naive_bayes as nb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt  

def ProcessData():
    f = open("car/car.data")
    data = np.genfromtxt(f, dtype={'names': ('buying', 'maint', 'doors','persons', 'lug_boot', 'safety', 'class'),
                                   'formats': ('S10', 'S10', 'S10','S10','S10', 'S10' ,'S10')}, delimiter=',')
    X=[]
    for i in ['buying', 'maint', 'doors','persons', 'lug_boot', 'safety']:
        le = LabelEncoder()
        le.fit(data[i])
        X.append(le.transform(data[i]))
    le = LabelEncoder()
    le.fit(data['class'])
    print le.classes_
    Y = (le.transform(data['class']))
    X = np.array(X).transpose()
    return (X,Y)




def Bayes(X_train, Y_train, X_test, Y_test):
    Bayes = nb.MultinomialNB()
    Bayes.fit(X_train,Y_train)
    e = 0
    pred = Bayes.predict(X_test)
    for i,j in izip(pred,Y_test):
        if i!=j:
            e = e + 1
    print e
    print X_train.shape
    score = Bayes.score(X_test,Y_test)
    print score
    plt.plot(range(Y_test.shape[0]), Y_test, 'bs', label='actual', markersize=7)
    plt.plot(range(Y_test.shape[0]), pred,'ro',  label='predicted', markersize=5)
    plt.title('Bayesian multivariate predictor (0 => acc, 1 =>good, 2=>unacc , 3=> vgood )')
    plt.text(0.95, 0.01, 'accuracy = '+ str(score))

    plt.legend(loc=4)
    plt.show()
    return score


(X,Y) = ProcessData()
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    Bayes(X[train], Y[train], X[test], Y[test])

