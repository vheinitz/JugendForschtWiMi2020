from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.model_selection import train_test_split

import dataLoader
import dataBuilder
from settings import BASE_PATH
from settings import TICKERS


def predict(X,Y):

    result = {}
    x_normalized = normalize(X, axis=0, norm='max')

    X_train, X_test, y_train, y_test = train_test_split(x_normalized, Y, test_size=0.25)

    # == Hier Algorithmus auswaehlen
    Classifiers = {
        #"svm linear c10" : svm.SVC(kernel='linear', C=10, gamma='auto')
        "svm sigmoid" : svm.SVC(kernel='sigmoid', C=10, gamma='auto')

    }
    for clf in Classifiers:
        #print(X_train)
        Classifiers[clf].fit(X_train, y_train)
        scr = Classifiers[clf].predict(X_test)
        result[clf]=scr
    print(result)


    return result

if __name__ == '__main__':

    stocks = dataLoader.load_stocks(BASE_PATH, TICKERS)

    for t in TICKERS:
        ticker = stocks[t]
        y = dataBuilder.get_diff_response_of(ticker)
        x = dataBuilder.get_many(stocks)
        yc = dataBuilder.diff_response_to_class(y)
        print (t)
        predict(x,yc)
