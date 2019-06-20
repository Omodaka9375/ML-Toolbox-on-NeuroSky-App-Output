import sys
import random
import csv
from itertools import izip_longest
import pspectrumlib as brainlib
import pickle as pk
import struct
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from featureselector import FeatureSelector
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import confusion_matrix

# General ML Algorithms toolbox for machine learning training and testing
# Author: Branislav Djalic
# Email: branislav.djalic@gmail.com

algo_list = ['dtc', 'linsvc', 'svc', 'mlp', 'knn', 'gaus', 'lda', 'logreg']

export_path = './model/'

# Data processing tools

def testAlgo(path='', samples=None, algo='', export=False, log=False, standard_scaler=False, minmax_scaler=False):

    if path == '' or algo == '':
        print('You need to specify all required parameters!')
        sys.exit(0)

    print('Training started . . .')
    X, y = extractDataset(path, row_count=samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=111, shuffle=True)

    if standard_scaler and not minmax_scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif minmax_scaler and not standard_scaler:
            mm = MinMaxScaler()
            X_train = mm.fit_transform(X_train)
            X_test = mm.transform(X_test)
    elif standard_scaler and minmax_scaler:
            print('You can only use one scaler at a time- minmax or standard!')
            sys.exit(0)

    if algo == 'dtc':
        model = DecisionTreeClassifier()
    if algo == 'linsvc':
        model = svm.LinearSVC()
    if algo == 'svc':
        model = svm.SVC()
    if algo == 'mlp':
        model = MLPClassifier()
    if algo == 'knn':
        model = KNeighborsClassifier()
    if algo == 'gaus':
        model = GaussianNB()
    if algo == 'lda':
        model = LinearDiscriminantAnalysis()
    if algo == 'logreg':
        model = LogisticRegression()

    model.fit(X_train, y_train)

    trainY = np.array(y_train)
    prediction_train = model.predict(X_train)

    testY = np.array(y_test)
    prediction_test = model.predict(X_test)

    training_accuracy = accuracy_score(trainY, prediction_train)
    test_accuracy = accuracy_score(testY, prediction_test)

    if export:
        export_trained_model(model, algo + '_classifier')
    return training_accuracy, test_accuracy


def testHypersOnAlgo(path='', samples=None, algo=[], hparameters={}, standard_scaler=False, minmax_scaler=False, folds=5, save_best=False, search='random'):

    if path == '' or len(algo) != len(hparameters) or len(hparameters) < 1:
        print('You need to specify all required parameters!')
        sys.exit(0)

    print('######### Hyper training started #########\n')

    X, y = extractDataset(path, row_count=samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=111, shuffle=True)

    if standard_scaler and not minmax_scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif minmax_scaler and not standard_scaler:
            mm = MinMaxScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif standard_scaler and minmax_scaler:
            print('You can only use one scaler at a time- minmax or standard!')
            sys.exit(0)

    best_score = {}

    if 'dtc' in hparameters:
                tree = DecisionTreeClassifier()

                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['dtc'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['dtc'], cv=folds)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                par = "Tuned DTC model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('DTC confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'best_dtc')

    if 'linsvc' in hparameters:
                tree = svm.LinearSVC()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['linsvc'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['linsvc'], cv=folds)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                par = "Tuned LinearSVC model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('LinearSVC confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'best_linsvc')

    if 'svc' in hparameters:
                tree = svm.SVC()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['svc'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['svc'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned SVC model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('SVC confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'best_svc')

    if 'mlp' in hparameters:
                tree =  MLPClassifier()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['mlp'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['mlp'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned MLP model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('MLP confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'best_mlp')

    if 'knn' in hparameters:
                tree = KNeighborsClassifier()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['knn'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['knn'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned KNN model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('KNN confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'knn_dtc')

    if 'gaus' in hparameters:
                tree = GaussianNB()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['gaus'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['gaus'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par ="Tuned Gaussian model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('Gaussian confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'gaus_dtc')

    if 'lda' in hparameters:
                tree = LinearDiscriminantAnalysis()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['lda'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['lda'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned LDA model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('LDA confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'lda_dtc')

    if 'logreg' in hparameters:
                tree = LogisticRegression()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['logreg'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['logreg'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned LogReg model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('Logreg confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'logreg_dtc')
    print('\n###### Best test score: ######')
    print(str(max(best_score.items(), key=lambda k: k[1])).replace('()','')+ '\n')    
    return best_score


def run_multiple(path='', algos=[], sample_count=None, log=False):
    print('\n########## Multi-algorithm testing started ##########\n')
    train_score = {}
    test_score = {}
    for i in range(len(algos)):
        training_accuracy, test_accuracy = testAlgo(path=path, algo=algos[i], samples=sample_count, export=False, log=False)
        train_score.update({algos[i]: training_accuracy})
        test_score.update({algos[i]: test_accuracy})
        print('########## ' + algos[i] + ' ##########')
        print("Train set score: {0} %".format(training_accuracy*100))
        print("Test set score: {0} %".format(test_accuracy*100)+ "\n")
    print('\nBest training score: ' + str(max(train_score.items(), key=lambda k: k[1])))
    print('Best test score: '+ str(max(test_score.items(), key=lambda k: k[1])))

def plot(path, sample_size=None, target=[], id='', kind=''):
      df = pd.read_csv(path,nrows=sample_size)
      df.groupby(target)[id].size().unstack().plot(kind=kind,stacked=True)
      plt.show()

def analyze(path, sample_count=None, save=False):
       df= pd.read_csv(path, sep = ",", nrows=sample_count, low_memory=False)
       print('Starting analysis . . .\n')
       print('Dataframe has shape: ' + str(df.shape))

       print('\nIdentifing bad features:\n')

       X= df.drop(df.columns[-1], axis='columns')
       y= df[df.columns[-1]]
       fs = FeatureSelector(data = X, labels = y)
       fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'regression', 'eval_metric': 'auc', 
                                     'cumulative_importance': 0.99})

       fs.plot_feature_importances(threshold = 0.99, plot_n = 15)

       if save:
               print('\nRemoving all bad features...')
               df = fs.remove(methods = 'all', keep_one_hot = False)
               print('Clean Dataframe has shape: ' + str(df.shape))
               print(df.head)
               df.to_csv('./clean_data/fe_clean_data_to_train.csv')
          
       return None

def extractDataset(path, row_count, log=False):
    df= pd.read_csv(path, nrows=row_count, low_memory=False)

    inputs = df.drop(df.columns[-1], axis='columns')
    target = df[df.columns[-1]]
    feature_list = list(inputs.columns)
    #feature_list = [float(k.strip('[] ')) for k in feature_list.split(',')]   
    for i in range(len(feature_list)):
        names = LabelEncoder()
        inputs[feature_list[i] + '_n'] = names.fit_transform(inputs[feature_list[i]])

    inputs_n = inputs.drop(feature_list, axis='columns')
    if log:
        print('Features head:' + inputs_n.head())
        print('Target ' + target.head())

    return inputs_n,target

def encodetest(path, row_count, log=False):
    df= pd.read_csv(path, sep = ",", nrows=row_count, low_memory=False)
    feature_list = list(inputs.columns)
    #feature_list = [float(k.strip('[] ')) for k in feature_list.split(',')]   
    for i in range(len(feature_list)):
        names = LabelEncoder()
        inputs[feature_list[i] + '_n'] = names.fit_transform(inputs[feature_list[i]])

    inputs_n = inputs.drop(feature_list, axis='columns')
    if log:
        print('Features head:' + inputs_n.head())
    return inputs_n

def export_trained_model(model, name):
    import pickle as pk
    path = export_path + name + '.csv'
    pk.dump(model, open(path, 'wb'))
    print('\nModel ' + name + ' saved at ' + path + '\n')

def predict_on_model(path, row_count, log=False):
    # load the model from disk and predict
    loaded_model = pk.load(open(path, 'rb'))
    X = encodetest('./clean_data/clean_data_to_train.csv',row_count=None, log=False)
    result = loaded_model.predict([X])
    print ('Model: ' + path + ' prediction: ' + result)

def extractTrainData(path, savepath, columns=[], row_count=None):
      print('\n########## ML toolbox starting ##########\n')
      print('Extracting training data...\n')
      
      if savepath == '' or len(columns) <1 or path == '':
            print('You need to specify all required parameters!')
            sys.exit(0)

      train_list = []
      for i in range (len(columns)):
            train_list.append(pd.read_csv(path, sep = ",",header=0, nrows=row_count, low_memory=False) [columns[i]])

      export_train_data = izip_longest(*train_list, fillvalue = '')
      with open(savepath, 'w') as myfile:
                  wr = csv.writer(myfile)
                  wr.writerow(columns)
                  wr.writerows(export_train_data)
      myfile.close()
      print('\nRaw train data extracted and cleaned at ' + savepath + '\n')

def extractTestData(path, savepath, columns=[], row_count=None):
      print('\nExtracting testing data . . .')

      if savepath == '' or len(columns) <1 or path == '':
            print('You need to specify all required parameters!')
            sys.exit(0)

      test_list=[]
      for i in range (len(columns)):   
            test_list.append(pd.read_csv(path, sep = ",",header=0, nrows=row_count, low_memory=False) [columns[i]])
      
      export_test_data = izip_longest(*test_list, fillvalue = '')
      with open(savepath, 'w') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(columns)
            wr.writerows(export_test_data)
      myfile.close()
      print('Raw test data extracted and cleaned at ' + savepath)

# Algorithms

# def autoregressione_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.ar_model import AR
    
#     # shuffle our data use 80:20 for train:test
#     random.shuffle(train_dataset)
#     n = int(len(train_dataset)*.80)

#     # create train and test cases
#     trainData = train_dataset[:n]
#     testData = train_dataset[n:]
#     # fit model
#     model = AR(trainData)
#     model_fit = model.fit()
#     # make prediction
#     train_prediction = model_fit.predict(len(trainData), len(trainData))
#     test_prediction = model_fit.predict(len(testData), len(testData))
#     print('Auto-regression results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'autoregressione_classifier')

# def movingaveragee_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.arima_model import ARMA
#     from random import random
#     # fit model
#     model = ARMA(train_dataset, order=(0, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Moving-average results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'movingaveragee_classifier')

# def ARmovingaveragee_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.arima_model import ARMA
#     from random import random
#     # fit model
#     model = ARMA(data, order=(2, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Autoregressive Moving Average (ARMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'ARmovingaveragee_classifier')

# def ARImovingaveragee_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.arima_model import ARIMA
#     from random import random
#     # fit model
#     model = ARIMA(data, order=(1, 1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset), typ='levels')
#     print('Autoregressive Moving Integrated Average (ARMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'ARImovingaveragee_classifier')

# def sarima_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     from random import random
#     # fit model
#     model = SARIMAX(train_dataset, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Seasonal Autoregressive Integrated Moving-Average (SARIMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'sarima_classifier')

# def sarimax_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     from random import random
#     # contrived dataset
#     train_dataset = [x + random() for x in range(1, 100)]
#     data2 = [x + random() for x in range(101, 200)]
#     # fit model
#     model = SARIMAX(train_dataset, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     exog2 = [200 + random()]
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset),exog=[exog2])
#     print('Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors(SARIMAX) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'sarimax_classifier')

# def vector_autoregressione_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.vector_ar.var_model import VAR
#     from random import random
#     # contrived dataset with dependency
#     train_dataset = list()
#     for i in range(100):
#         v1 = i + random()
#         v2 = v1 + random()
#         row = [v1, v2]
#         train_dataset.append(row)
#     # fit model
#     model = VAR(train_dataset)
#     model_fit = model.fit()
#     # make prediction
#     prediction = model_fit.forecast(model_fit.y, steps=1)
#     print('Vector Autoregression (VAR) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'vector_autoregressione_classifier')

# def vector_autoregression_movingavr_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.varmax import VARMAX
#     from random import random
#     # contrived dataset with dependency
#     train_dataset = list()
#     for i in range(100):
#         v1 = random()
#         v2 = v1 + random()
#         row = [v1, v2]
#         train_dataset.append(row)
#     # fit model
#     model = VARMAX(train_dataset, order=(1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.forecast()
#     print('Vector Autoregression Moving-Average (VARMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'vector_autoregression_movingavr_classifier')

# def varmaxe_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.varmax import VARMAX
#     from random import random
#     # contrived dataset with dependency
#     train_dataset = list()
#     for i in range(100):
#         v1 = random()
#         v2 = v1 + random()
#         row = [v1, v2]
#         train_dataset.append(row)
#     data_exog = [x + random() for x in range(100)]
#     # fit model
#     model = VARMAX(train_dataset, exog=data_exog, order=(1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     data_exog2 = [[100]]
#     prediction = model_fit.forecast(exog=data_exog2)
#     print('Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'varmaxe_classifier')

# def simple_expo_smoothinge_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.holtwinters import SimpleExpSmoothing
#     # fit model
#     model = SimpleExpSmoothing(train_dataset)
#     model_fit = model.fit()
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Simple Exponential Smoothing (SES) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'simple_expo_smoothinge_classifier')

# def holtwintere_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.holtwinters import ExponentialSmoothing
#     # fit model
#     model = ExponentialSmoothing(train_dataset)
#     model_fit = model.fit()
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Holt Winters Exponential Smoothing (HWES) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'holtwintere_classifier')


##### brain spectrum converter for EEG
from scipy.interpolate import interp1d
import statistics
from scipy.stats import kurtosis
from scipy.stats import skew

def pSpectrum(vector):
    '''get the power spectrum of a vector of raw EEG data'''
    A = np.fft.fft(vector)
    ps = np.abs(A)**2
    ps = ps[:len(ps)//2]
    return ps
  
def entropy(power_spectrum,q):
    '''get the entropy of a power spectrum'''
    q = float(q)
    
    power_spectrum = np.array(power_spectrum)
        
    if not q ==1:
        S = 1/(q-1)*(1-np.sum(power_spectrum**q))
    else:
        S = - np.sum(power_spectrum*np.log2(power_spectrum))
        
    return S

def binnedPowerSpectra (pspectra,nbin):
    '''compress an array of power spectra into vectors of length nbins'''
    l = len(pspectra)
    array = np.zeros([l,nbin])

    for i,ps in enumerate(pspectra):
        x = np.arange(1,len(ps)+1)
        f = interp1d(x,ps)#/np.sum(ps)
        array[i] = f(np.arange(1, nbin+1))

    index = np.argwhere(array[:,0]==-1)
    array = np.delete(array,index,0)
    return array

# get the power spectrum
def spectra (readings):
  "Parse + calculate the power spectrum for every reading in a list"
  return [pSpectrum(v) for v in readings]

def avgPowerSpectrum (arrayOfPowerSpectra, modifierFn):
    '''
    get the mean of an array of power spectra, and apply modifierFn to it
    example: 
    avgPowerSpectrum(binnedPowerSpectra(pspectra,100), np.log10)
    '''
    # ra = modifierFn(np.mean(arrayOfPowerSpectra, 0))
    # return  np.array_str(ra, max_line_width=np.inf)
    return modifierFn(np.mean(arrayOfPowerSpectra, 0))

def avgPercentileUp (arrayOfPowerSpectra, confidenceIntervalParameter):
    '''confidenceIntervalParameter of 1 is 1%-99%'''
    return np.percentile(spectra,100-confidenceIntervalParameter,axis=0)

def avgPercentileDown (arrayOfPowerSpectra, confidenceIntervalParameter):
    return np.percentile(spectra,confidenceIntervalParameter,axis=0)

def pinkNoiseCharacterize(pspectrum,normalize=True,plot=True):
    '''Compute main power spectrum characteristics'''
    if normalize:
        pspectrum = pspectrum/np.sum(pspectrum)
    
    S = entropy(pspectrum,1)
    
    x = np.arange(1,len(pspectrum)+1)
    lx = np.log10(x)
    ly = np.log10(pspectrum)
    
    c1 = (x > 0)*(x < 80)
    c2 = x >= 80
    
    fit1 = stats.linregress(lx[c1],ly[c1])
    fit2 = stats.linregress(lx[c2],ly[c2])
    
    #print fit1
    #print fit2
    
    if plot:
        plot(lx,ly)
        plot(lx[c1],lx[c1]*fit1[0]+fit1[1],'r-')
        plot(lx[c2],lx[c2]*fit2[0]+fit2[1],'r-')
        
    return {'S':S,'slope1':fit1[0],'slope2':fit2[0]}

# A function we apply to each group of power spectra
def makeFeatureVector (readings, bins): 
  '''
  Create 100, log10-spaced bins for each power spectrum.
  For more, see http://blog.cosmopol.is/eeg/2015/06/26/pre-processing-EEG-consumer-devices.html
  '''
  return avgPowerSpectrum(
    binnedPowerSpectra(spectra(readings), bins)
    , np.log10)

def extractFeatures(ll):
    ss = []
    max_value = max(ll)
    min_value = min(ll)
    standard_dev = statistics.stdev(ll)
    sk = skew(ll)
    ku = kurtosis(ll)
    ss.append(max_value)
    ss.append(min_value)
    ss.append(standard_dev)
    ss.append(sk)
    ss.append(ku)
    return ss

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

testdataset = []
def create_dataset(arr, label):
    for x in arr:
        testdataset.append([x, label])

def multiToModel(labels, recordings, examples,raw_value_name):
    print('####### Session started #######\n')
    print('Going through raw data:\n')
    for t in range(len(labels)):
        for recording in range(recordings):
            fArray = []
            read_path = './raw_data/'+ str(labels[t]) + '/' + str(recording) + '.csv'
            print(read_path)
            raw_value_per_person = pd.read_csv(read_path, sep = ",", nrows=15300) [raw_value_name] ##1020 is 2 sec // 2500 is 5 sec
            m = raw_value_per_person.to_json(orient = 'records') 
            sempl = [int(k.strip('[] ')) for k in m.split(',')]
            all_samples = brainlib.split_list(sempl,wanted_parts=examples)    
            for i in range(len(all_samples)):
                feature = []
                # if(feature_representation == 0):
                feature = brainlib.makeFeatureVector([all_samples[i]],100) #convert raw data to power spec vectors  
                # if(feature_representation == 1):
                #     feature = brainlib.extractFeatures(all_samples[i]) # use min.max.std,skew and ku
                fArray.append(feature) #each feature is 5 sec rec
            create_dataset(fArray, t)
    df = pd.DataFrame(testdataset)
    df.to_csv('./clean_data/clean_data_to_train.csv')