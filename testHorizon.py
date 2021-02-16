import numpy as np
from numpy import std
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import svm
import dataLoader
import dataBuilder
from settings import BASE_PATH
from settings import TICKERS
from settings import ClassIdUp
from settings import ClassIdDown
from settings import MODEL_ROW_TICKS
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
from settings import ClassIdUp
from settings import ClassIdDown
from settings import MODEL_ROW_TICKS
import pickle
from joblib import dump, load

def test_classifier(X,Y):

    result = {}
    x_normalized = normalize(X, axis=0, norm='max')

    X_train, X_test, y_train, y_test = train_test_split(x_normalized, Y, test_size=0.25)

    # == Hier Algorithmus auswaehlen
    Classifiers = {
        #"svm rbf" : svm.SVC(kernel='rbf', C=1, gamma='auto')
        #,"svm linear" : svm.SVC(kernel='linear', C=1)
        "svm_linear_c10" : svm.SVC(kernel='linear', C=10, gamma='auto')
        #,"svm poly" : svm.SVC(kernel='poly', C=10, gamma='auto')
        #,"svm sigmoid" : svm.SVC(kernel='sigmoid', C=10, gamma='auto')
        ,"svm_flinear_c10": svm.SVC(kernel='linear', C=10, gamma='auto')
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
    import os.path
    # == Hier wird gelernt
    for clf in Classifiers:
        if os.path.isfile(clf):
            Classifiers[clf] = load(clf)
        else:
            print("File exists")
            Classifiers[clf].fit(X_train, y_train)
            scr = Classifiers[clf].score(X_test, y_test)
            result[clf]=scr
            dump(Classifiers[clf], clf)
            print(clf,scr)

    #print(result)
    return result

def passive_result(y,ticker):
    ClassIdUpCnt = y.count(ClassIdUp)
    ClassIdDownCnt = y.count(ClassIdDown)
    return(ClassIdUpCnt / (ClassIdDownCnt + ClassIdUpCnt))

def analyze_result(x,y, alignClassNumber=False, tickerName="", classifier=""):

    ClassIdUpCnt = y.count(ClassIdUp)
    ClassIdDownCnt = y.count(ClassIdDown)

    maxNumOfInstances = min(ClassIdUpCnt,ClassIdDownCnt)

    if alignClassNumber:
        x1=[]
        y1=[]
        cntUp=0
        cntDown=0
        for xi, yi in zip(x, y):

            if yi == ClassIdUp and cntUp < maxNumOfInstances:
                if len(x1) == 0:
                    x1 = xi
                else:
                    x1 = np.vstack([x1, xi])
                y1.append(yi)
                cntUp+=1
            if yi == ClassIdDown and cntDown < maxNumOfInstances:
                if len(x1) == 0:
                    x1 = xi
                else:
                    x1 = np.vstack([x1, xi])
                y1.append(yi)
                cntDown+=1

        x=x1
        y=y1

    ClassIdUpCnt = y.count(ClassIdUp)
    ClassIdDownCnt = y.count(ClassIdDown)

    classifiers_result = test_classifier(x,y)
    N=4
    for i in range(0,N):
        tmp = test_classifier(x,y)
        #print("Iteration N: %d" %(i))
        for k in tmp:
            classifiers_result[k]+= tmp[k]

    for k in classifiers_result:
        classifiers_result[k] /= (N+1)
    return(classifiers_result[classifier])



if __name__ == '__main__':

    stocks = dataLoader.load_stocks(BASE_PATH, TICKERS)
    resultsl = []
    #resultss = []
    #fiftyline = []
    resultsf = []
    #resultsn = []
    #resultsp = []
    for d in range(30):
        print(d)
        ticker = stocks["COST"]
        y = dataBuilder.get_diff_response_of(ticker,futureTicks=d)
        x = dataBuilder.get_many(stocks)
        yc = dataBuilder.diff_response_to_class(y)
        resultsl.append(analyze_result(x[0:len(y)], yc, alignClassNumber=True, tickerName="COST", classifier='svm_linear_c10'))
        #resultss.append(analyze_result(x[0:len(y)], yc, alignClassNumber=True, tickerName="GOOG", classifier='svm sigmoid'))
        resultsf.append(analyze_result(x[0:len(y)], yc, alignClassNumber=False, tickerName="GOOG", classifier='svm_flinear_c10'))
        #resultsn.append(analyze_result(x[0:len(y)], yc, alignClassNumber=True, tickerName="GOOG", classifier='mlpc'))
        #resultsp.append(passive_result(yc,t))
    print(sum(resultsl)/30)
    print(np.std([resultsl]))
    #print(sum(resultsn)/30)
    #print(np.std([resultsn]))
    #print(sum(resultss)/30)
    #print(np.std([resultss]))
    #print(sum(resultsp)/30)
    #print(np.std([resultsp]))
    print(sum(resultsf)/30)
    print(np.std([resultsf]))
    #print(TICKERS)
    #plt.plot(resultsl, "bo", fiftyline, "r--",resultsf, "y*",resultsp,"g+",resultss,"m^",resultsn,"c>")
    plt.plot(resultsl, "bo",resultsf,"g>")
    plt.show()

