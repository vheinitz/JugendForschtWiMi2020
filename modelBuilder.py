import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

import dataLoader
import dataBuilder
from settings import BASE_PATH
from settings import TICKERS
from settings import MODEL_ROW_TICKS


def test_classifier(X,Y):

    result = {}
    x_normalized = normalize(X, axis=0, norm='max')

    X_train, X_test, y_train, y_test = train_test_split(x_normalized, Y, test_size=0.25)

    # == Hier Algorithmus auswaehlen
    Classifiers = {
        #"svm rbf" : svm.SVC(kernel='rbf', C=1, gamma='auto')
        #,"svm linear" : svm.SVC(kernel='linear', C=1)
        "svm linear c10" : svm.SVC(kernel='linear', C=10, gamma='auto')
        #,"svm poly" : svm.SVC(kernel='poly', C=10, gamma='auto')
        ,"svm sigmoid" : svm.SVC(kernel='sigmoid', C=10, gamma='auto')
        #,"adaboost" : AdaBoostClassifier()
        #,"mlpc" : MLPClassifier(alpha=1, max_iter=1000)
        #,"dec tree" : DecisionTreeClassifier(max_depth=15)
        #,"gaussian" : GaussianNB()
        #,"rfc 5 10 1" : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        #,"rfc 5 10 5" : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=5)
        #,"rfc 7 10 1" : RandomForestClassifier(max_depth=7, n_estimators=10, max_features=1)
        #,"rfc 10 10 2" : RandomForestClassifier(max_depth=10, n_estimators=10, max_features=2)
        #,"rfc 5 100 5" : RandomForestClassifier(max_depth=5, n_estimators=100, max_features=5)
    }

    # == Hier wird gelernt
    for clf in Classifiers:
        Classifiers[clf].fit(X_train, y_train)
        scr = Classifiers[clf].score(X_test, y_test)
        result[clf]=scr
        #print(clf,scr)
    #print(result)
    return result


def analyze_classifier(x,y):

    classifiers_result = test_classifier(x,y)
    N=10
    for i in range(0,N):
        tmp = test_classifier(x,y)
        #print("Iteration N: %d" %(i))
        for k in tmp:
            classifiers_result[k]+= tmp[k]

    for k in classifiers_result:
        classifiers_result[k] /= (N+1)
    print(classifiers_result)



if __name__ == '__main__':

    stocks = dataLoader.load_stocks(BASE_PATH, TICKERS)

    for t in TICKERS:
        ticker = stocks[t]
        y = dataBuilder.get_diff_response_of(ticker)
        x = dataBuilder.get_many(stocks)
        yc = dataBuilder.diff_response_to_class(y)
        print (t)
        analyze_classifier(x,yc)
