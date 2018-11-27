#!!! Image prediction algorithms
# Linear SVC, SVC with RBF kenel, Random forest, XGBoost, Logistic rergression, Naive Bayes
 
import numpy as np
import pandas as pd
from sklearn import svm
import h2o
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
import utils

def getScore(y_true, y_pred):
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return None
    y = np.concatenate((y_true, y_pred))
    classLabels = np.unique(y) # sorted array
    index = []
    columns = []
    for label in classLabels:
        index.append('{}{}'.format('Predicted ', label))
        columns.append('{}{}'.format('Actual ', label))
    cm = np.zeros(shape=(len(classLabels), len(classLabels)), dtype=int)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.ndim == 2:
        y_true = y_true.reshape(-1)
    if y_pred.ndim == 2:
        y_pred = y_pred.reshape(-1)
    if len(y_true) != len(y_pred):
        return False
    nTotal = len(y_true)
    for i in range(0, len(y_true), 1):
        y_trueValue = y_true[i]
        y_predValue = y_pred[i]
        row = None
        column = None
        for idx, label in enumerate(classLabels, start=0):# tuples[index, label]
            if label == y_trueValue:
                column = idx
            if label == y_predValue:
                row = idx
        if row is None or column is None:
            print("Crash getScore. Class does not contain label")
            return False
        cm[row, column] += 1 
                    
    cmFrame = pd.DataFrame(cm, index=index, columns=columns, dtype=int)
# sum of all non-diagonal cells of cm / nTotal        
    misclassification = 0
    accuracy = 0
    for i in range(0, len(classLabels), 1):
        for j in range(0, len(classLabels), 1):
            if i == j:
                accuracy += cm[i, j]
                continue
            misclassification += cm[i, j]
    misclassification /= nTotal  
    accuracy /= nTotal 
    out = {}
    for i in range(0, len(classLabels), 1):
        out[classLabels[i]] = {}
        tp_fp = np.sum(cm[i, :])
        tp_fn = np.sum(cm[:, i])
        if tp_fp != 0:
            out[classLabels[i]]['Precision'] = cm[i, i] / tp_fp # tp / (tp + fp)            
        else:
            out[classLabels[i]]['Precision'] = 0
        if tp_fn != 0:            
            out[classLabels[i]]['Recall'] = cm[i, i] / tp_fn  # tp / (tp + fn)    
        else:
            out[classLabels[i]]['Recall'] = 0
        if (out[classLabels[i]]['Precision'] + out[classLabels[i]]['Recall']) != 0:
            out[classLabels[i]]['F1'] = 2 * (out[classLabels[i]]['Precision'] * \
                out[classLabels[i]]['Recall']) / (out[classLabels[i]]['Precision'] + \
                out[classLabels[i]]['Recall'])
        else:
            out[classLabels[i]]['F1'] = 0

    return {'Accuracy': accuracy, 'Misclassification': misclassification,\
            'Confusion_Matrix': cmFrame, 'Labels': out}

def validateFit(estimator, data, columnsX, columnY, nFolds=5, Labels=None):     

    setPure = data.copy(deep=True)
    setPure.reset_index(drop=True, inplace=True) # reset index
    setPure = setPure.reindex(np.random.permutation(setPure.index)) # shuffle 
    setPure.sort_index(inplace=True)
        
    size = setPure.shape[0]
    if size < nFolds:
        return None# too few observations
        
    binSize = int(size / nFolds)
    lo = 0
    hi = binSize-1
    intervals = []
    i = 1
    while True:
        i += 1
        intervals.append((lo, hi))
        if hi == (size-1):
            break
        lo += binSize
        hi += binSize
        if i == nFolds:
            hi = size-1        
    
    scoresValid = []
    for i in range(0, len(intervals), 1):   
        print("Split ", i)
        lo = intervals[i][0]
        hi = intervals[i][1]
            
        setValid = setPure.loc[lo:hi, :]
        intervalsTrain = copy.deepcopy(intervals)
        del(intervalsTrain[i]) # remove interval currently using for test set
        setTrain = None
        # train test split        
        for j in range(0, len(intervalsTrain), 1):   
            low = intervalsTrain[j][0]
            high = intervalsTrain[j][1]
            if setTrain is None:
                setTrain = setPure.loc[low:high, :].copy(deep=True)
            else:
                new = setPure.loc[low:high, :].copy(deep=True)
                setTrain = pd.concat([setTrain, new], axis=0)
        
        scoreValid = None
        if estimator.__class__.__name__ == 'H2ODeepLearningEstimator': # check it
            
            trainH2O = h2o.H2OFrame(setTrain)
            validH2O = h2o.H2OFrame(setValid)
    
            estimator.train(x = columnsX, y = columnY, training_frame = trainH2O)
                            
            y_predH2O = estimator.predict(validH2O)
            y_pred = h2o.as_list(y_predH2O)
            y_pred = np.where(y_pred['predict'] == 'Yes', 1, 0)
            y_true_valid = np.where(setValid[columnY] == 'Yes', 1, 0)
            scoreValid = getScore(y_true_valid, y_pred)
                          
        elif estimator.__class__.__name__ == 'RandomForestClassifier' or \
            estimator.__class__.__name__ == 'XGBClassifier' or \
            estimator.__class__.__name__ == 'SVC' or \
            estimator.__class__.__name__ == 'LogisticRegression' or \
            estimator.__class__.__name__ == 'MultinomialNB':
                
                
            y_true_train = setTrain[columnY].values
            estimator.fit(setTrain.loc[:, columnsX].values, y_true_train)
            
            y_true_valid = setValid[columnY].values
            y_pred = estimator.predict(setValid.loc[:, columnsX].values)            
            scoreValid = getScore(y_true_valid, y_pred)
            
        scoresValid.append(scoreValid)
        
    if len(scoresValid) > 0:
        accuracy = 0
        misclassification = 0
        F1 = {}
        for score in scoresValid:
            accuracy += score['Accuracy']
            misclassification += score['Misclassification']
            F1 = {}
            for key, value in score['Labels'].items():
                if key in F1.keys():
                    F1[key] += score['Labels'][key]['F1']
                else:
                    F1[key] = score['Labels'][key]['F1']
        accuracy /= len(scoresValid)
        misclassification /= len(scoresValid)
    # average F1        
    F1Avg = 0
    for f1 in F1.values():
        F1Avg += f1
    F1Avg /= len(F1)

    return {'Accuracy': accuracy, 'Misclassification': misclassification,\
            'F1': F1, 'F1_Average': F1Avg, 'Folds': scoresValid}

def fitPredict(estimator, dataTrain, dataForecast, columnsX, columnY):
    yTrain = dataTrain[columnY].values
    estimator.fit(dataTrain.loc[:, columnsX].values, yTrain)            
    yForecast = estimator.predict(dataForecast.loc[:, columnsX].values)            
    return yForecast

def reshapeImages(images):
    x = np.zeros(shape=(len(images), len(images[0, 1])), dtype=np.uint8)
    for i in range(0, len(images), 1):
        x[i, :] = images[i, 1]
    return x

algorithms = ['SVClinears', 'SVC', 'RF', 'XG', 'NB']
#Load images with numpy
imagesTrain = np.load('train_images32x32.npy', encoding='latin1')
imagesForecast = np.load('test_images32x32.npy', encoding='latin1')

labelsTrain = pd.read_csv('train_labels_new.csv')
labelsTrain.drop(columns=['Id'], inplace=True)

labelsForecast = pd.DataFrame(index=range(0, len(imagesForecast), 1),\
    columns=algorithms)

xTrain = reshapeImages(imagesTrain)
xForecast = reshapeImages(imagesForecast)

labelencoder_X=LabelEncoder()
labelsTrain['Numeric'] = labelencoder_X.fit_transform(labelsTrain.Category.values)

yTrain = labelsTrain['Numeric'].values.astype(np.uint8)

columnsTrain = list(range(0, len(imagesTrain[0, 1]), 1))
for i in range(0, len(imagesTrain[0, 1]), 1):
    columnsTrain[i] = str(columnsTrain[i])

columnsTrain.append('Y')

dataTrain = pd.DataFrame(np.concatenate((xTrain, yTrain.reshape(-1, 1)), axis=1), columns=columnsTrain)
del(columnsTrain[-1])

dataForecast = pd.DataFrame(xForecast, columns=columnsTrain)


# linear svc
print("Linear SVC") # 13%
classifierSVClinear = svm.SVC(C=10.0, kernel='linear', gamma='auto',\
    coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,\
    class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',\
    random_state=None)

scoreSVClinear = validateFit(classifierSVClinear, dataTrain, columnsTrain, 'Y', nFolds=5, Labels=None)

y_SVClinear = fitPredict(classifierSVClinear, dataTrain, dataForecast, columnsTrain, 'Y')
labelsSVClinears = list(labelencoder_X.inverse_transform(y_SVClinear))
labelsForecast['SVClinears'] = labelsSVClinears

# rbf svc
print("RBF SVC") # 22%
classifierSVC = svm.SVC(C=10.0, kernel='rbf', degree=3, gamma='auto',\
    coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,\
    class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',\
    random_state=None)

scoreSVC = validateFit(classifierSVC, dataTrain, columnsTrain, 'Y', nFolds=5, Labels=None)

y_SVC = fitPredict(classifierSVC, dataTrain, dataForecast, columnsTrain, 'Y')
labelsSVC = list(labelencoder_X.inverse_transform(y_SVC))
labelsForecast['SVC'] = labelsSVC

print("Random Forest") # 13%
classifierRF = RandomForestClassifier(n_estimators=10, criterion='entropy',\
    max_depth=3, min_samples_split=20, min_samples_leaf=1,\
    min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None,\
    min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,\
    oob_score=False, warm_start=False, class_weight=None)

scoreRF = validateFit(classifierRF, dataTrain, columnsTrain, 'Y', nFolds=5, Labels=None)

y_RF = fitPredict(classifierRF, dataTrain, dataForecast, columnsTrain, 'Y')
labelsRF = list(labelencoder_X.inverse_transform(y_RF))
labelsForecast['RF'] = labelsRF

# XGBoost
print("XGBoost") # 13%
classifierXG = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=10,\
    silent=False, objective='multi:softprob', booster='gbtree', \
    n_jobs=1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,\
    colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,\
    scale_pos_weight=1, base_score=0.5, seed=None, missing=None)

scoreXG = validateFit(classifierXG, dataTrain, columnsTrain, 'Y', nFolds=5, Labels=None)

y_XG = fitPredict(classifierXG, dataTrain, dataForecast, columnsTrain, 'Y')
labelsXG = list(labelencoder_X.inverse_transform(y_XG))
labelsForecast['XG'] = labelsXG

# 10%
print("MultinomialNB") # only 1 core
classifierNB = MultinomialNB(alpha=1.0)

scoreNB = validateFit(classifierNB, dataTrain, columnsTrain, 'Y', nFolds=5, Labels=None)

y_NB = fitPredict(classifierNB, dataTrain, dataForecast, columnsTrain, 'Y')
labelsNB = list(labelencoder_X.inverse_transform(y_NB))
labelsForecast['NB'] = labelsNB

summary = pd.DataFrame(index=algorithms, columns=['F1_Average', 'Accuracy'])

summary.loc['SVClinears', 'F1_Average'] = scoreSVClinear['F1_Average']
summary.loc['SVClinears', 'Accuracy'] = scoreSVClinear['Accuracy']

summary.loc['SVC', 'F1_Average'] = scoreSVC['F1_Average']
summary.loc['SVC', 'Accuracy'] = scoreSVC['Accuracy']

summary.loc['RF', 'F1_Average'] = scoreRF['F1_Average']
summary.loc['RF', 'Accuracy'] = scoreRF['Accuracy']

summary.loc['XG', 'F1_Average'] = scoreXG['F1_Average']
summary.loc['XG', 'Accuracy'] = scoreXG['Accuracy']

summary.loc['NB', 'F1_Average'] = scoreNB['F1_Average']
summary.loc['NB', 'Accuracy'] = scoreNB['Accuracy']

utils.saveObject('labelsForecast.dat', labelsForecast)
utils.saveObject('summary.dat', summary)

writer = pd.ExcelWriter('summary.xlsx')
summary.to_excel(writer,'Sheet1')
writer.save()

writer = pd.ExcelWriter('labelsForecast.xlsx')
labelsForecast.to_excel(writer,'Sheet1')
writer.save()

