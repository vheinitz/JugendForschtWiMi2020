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
stocks = dataLoader.load_stocks(BASE_PATH, TICKERS)
ticker = stocks["GOOG"]
y = dataBuilder.get_diff_response_of(ticker)
yc = dataBuilder.diff_response_to_class(y)
print(yc)
