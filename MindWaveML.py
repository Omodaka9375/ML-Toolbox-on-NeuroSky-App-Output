import pandas as pd
import pickle as pk
import numpy as np
from sklearn import svm
from sklearn import model_selection 
import pspectrumlib as brainlib
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import random as random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# labels
labels = ['apple','banana', 'music', 'orange', 'triangle']
# every cvs is one raw example
recordings = 4 # how many 30 sec recordings there are
extract_columns = ['eegRawValue']#, delta','theta','alphaLow','alphaHigh','betaLow','betaHigh','gammaLow','gammaMid']

feature_representation = 0 # ['power spectrum', 'mean']


# where to store the model
filename = './model/finalized_model.sav'
dataset = []

def create_dataset(arr, label):
    for x in arr:
        dataset.append([x, label])

def cvsToModel(labels):
    print('####### Session started #######\n')
    print('Going through raw data:\n')
    for t in range(len(labels)):
        for recording in range(recordings):
            fArray = []
            read_path = './raw_data/'+ str(labels[t]) + '/' + str(recording) + '.csv'
            print(read_path)
            df_ex = pd.read_csv(read_path, sep = ',', nrows=15360)  ##1024 is 2 sec // 2500 is 5 sec
            df = df_ex[extract_columns]
            print(df.shape)
            NUMBER_OF_SPLITS = 15
            for i, new_df in enumerate(np.array_split(df,NUMBER_OF_SPLITS)):
                print(new_df.shape)
                new_df = new_df.applymap(str)
                temp_vecs = new_df.values.tolist()
                result = [map(int, list(l)) for l in temp_vecs]
                flat_list = [item for sublist in result for item in sublist]
                all_samples = brainlib.split_list(flat_list,wanted_parts=4)
                feature = brainlib.makeFeatureVector(all_samples,100)
                #print(feature)
                fArray.append(feature) #each feature is 2 sec rec
            create_dataset(fArray, t)
    # dff = pd.DataFrame(dataset, dtype=int)
    # dff.to_csv('data.csv', header=False, index=False)
    train()
    
        
def train():
    # shuffle our data use 80:20 for train:test
    random.shuffle(dataset)
    n = int(len(dataset)*.75)

    # create train and test cases
    trainData = dataset[:n]
    testData = dataset[n:]

    # SVM machines
    clf = svm.SVC(kernel='rbf', verbose=0, random_state=1,probability=True)
    #clf = svm.LinearSVC(verbose = True, dual=False, multi_class='ovr', random_state=46, max_iter=100000, tol=0.00001, C=1.0)
                   
    # perceptron and DTC + lbfgs 'identity', 'logistic', 'tanh', 'relu'
    #clf = MLPClassifier()
    #clf = MLPClassifier(solver='sgd', learning_rate_init=0.001, activation="tanh", early_stopping=True, hidden_layer_sizes=(100,), warm_start=False, max_iter=2000, verbose=False)
    #clf = DecisionTreeClassifier()

    # normalizing data
    # scaler = MinMaxScaler() 

    # X = scaler.fit_transform([entry[0] for entry in trainData])
    # Y = [entry[1] for entry in trainData]

    # testX = scaler.fit_transform([entry[0] for entry in testData])
    # testY = [entry[1] for entry in testData]

    # without additional normalizing
    X = [entry[0] for entry in trainData]
    Y = [entry[1] for entry in trainData]

    testX = [entry[0] for entry in testData]
    testY = [entry[1] for entry in testData]

    # plot data
    #plt.scatter(X,Y)
    #plt.scatter(testX,testY)
    #plt.show()

    clf.fit(X,Y)

    # LDA example 
    # lda = LinearDiscriminantAnalysis()
    # lda.fit(X, Y)
    # print('Accuracy of LDA classifier on training set: {:.2f}'.format(lda.score(X, Y)))
    # print('Accuracy of LDA classifier on test set: {:.2f}'.format(lda.score(testX, testY)))

    # gaussian example
    # gnb = GaussianNB()
    # gnb.fit(X, Y)
    # print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X, Y)))
    # print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(testX, testY)))

    # KNN example - not working
    # from sklearn.metrics import classification_report
    # from sklearn.metrics import confusion_matrix
    # from sklearn.neighbors import KNeighborsClassifier
    # knn = KNeighborsClassifier(n_neighbors = np.arange(1,5))
    # knn.fit(X.all(),Y)
    # pred = knn.predict(testX)
    # print(confusion_matrix(testY, pred))
    # print(classification_report(testY, pred))

    trainY = np.array(Y)
    trainY_pred = clf.predict(X)

    testY = np.array(testY)
    testY_pred = clf.predict(testX)

    a1 = accuracy_score(trainY,trainY_pred)
    a2 = accuracy_score(testY,testY_pred)
    print("\nAccuracy train set: {0} %".format(a1*100))
    print("Accuracy test set: {0} %".format(a2*100))
    # save the model to disk
    #if a2 > 0.98:
    pk.dump(clf, open(filename, 'wb'))
    print('\nModel saved at ' + filename + '\n')
    

cvsToModel(np.asarray(labels))
#train()
