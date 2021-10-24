# Import libraries 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import numpy as np 
import pandas as pd 
import csv 

trainingInfo  = pd.read_csv('train_final.csv')
testingInfo = pd.read_csv('test_final.csv')

#print(trainingInfo.describe())

trainingDataFrame = pd.DataFrame(trainingInfo)
testingDataFrame = pd.DataFrame(testingInfo)
#print(trainingDataFrame.columns)

#print(trainingDataFrame.isin(['?']).sum(axis=0))
#print(testingDataFrame.isin(['?']).sum(axis=0))

#print(trainingDataFrame.columns)

#Process Data
trainingDataFrame['sex'] = trainingDataFrame['sex'].map(
    {
        'Female': 0, 
        'Male': 1
    }).astype(int)

trainingDataFrame['race'] = trainingDataFrame['race'].map(
    {
        'White': 0, 
        'Asian-Pac-Islander': 1, 
        'Amer-Indian-Eskimo': 2, 
        'Black': 3, 
        'Other': 4
    }).astype(int)

trainingDataFrame['marital.status'] = trainingDataFrame['marital.status'].map(
    {
        'Married-civ-spouse': 0, 
        'Divorced': 1, 
        'Never-married': 2, 
        'Separated': 3, 
        'Widowed': 4,
        'Married-spouse-absent': 5, 
        'Married-AF-spouse': 6
    }).astype(int)

trainingDataFrame['workclass'] = trainingDataFrame['workclass'].map(
    {
        'Private': 0, 
        'Self-emp-not-inc': 1,
        'Self-emp-inc': 2, 
        'Federal-gov': 3, 
        'Local-gov': 4,
        'State-gov': 5, 
        'Without-pay': 6,
        'Never-worked': 7,
        '?': 8
    }).astype(int)

trainingDataFrame['education'] = trainingDataFrame['education'].map(
    {
        'Bachelors': 0, 
        'Some-college': 1, 
        '11th': 2, 
        'HS-grad': 3, 
        'Prof-school': 4, 
        'Assoc-acdm': 5, 
        'Assoc-voc': 6, 
        '9th': 7,
        '7th-8th': 8, 
        '12th': 9, 
        'Masters': 10, 
        '1st-4th': 11,
        '10th': 12,
        'Doctorate': 13,
        '5th-6th': 14,
        'Preschool': 15
    }).astype(int)

trainingDataFrame['occupation'] = trainingDataFrame['occupation'].map(
    { 
        'Tech-support': 1, 
        'Craft-repair': 2, 
        'Other-service': 3, 
        'Sales': 4, 
        'Exec-managerial': 5,
        'Prof-specialty': 6, 
        'Handlers-cleaners': 7,
        'Machine-op-inspct': 8,
        'Adm-clerical': 9,
        'Farming-fishing': 10, 
        'Transport-moving': 11, 
        'Priv-house-serv': 12, 
        'Protective-serv': 13,
        'Armed-Forces': 14,
        '?': 15
    }).astype(int)

trainingDataFrame['relationship'] = trainingDataFrame['relationship'].map(
    {
        'Wife': 0, 
        'Own-child': 1, 
        'Husband': 2, 
        'Not-in-family': 3,
        'Other-relative': 4,
        'Unmarried': 5
    }).astype(int)

testingDataFrame['sex'] = testingDataFrame['sex'].map(
    {
        'Female': 0, 
        'Male': 1
    }).astype(int)

testingDataFrame['race'] = testingDataFrame['race'].map(
    {
        'White': 0, 
        'Asian-Pac-Islander': 1, 
        'Amer-Indian-Eskimo': 2, 
        'Black': 3, 
        'Other': 4
    }).astype(int)

testingDataFrame['marital.status'] = testingDataFrame['marital.status'].map(
    {
        'Married-civ-spouse': 0, 
        'Divorced': 1, 
        'Never-married': 2, 
        'Separated': 3, 
        'Widowed': 4,
        'Married-spouse-absent': 5, 
        'Married-AF-spouse': 6
    }).astype(int)

testingDataFrame['workclass'] = testingDataFrame['workclass'].map(
    {
        'Private': 0, 
        'Self-emp-not-inc': 1,
        'Self-emp-inc': 2, 
        'Federal-gov': 3, 
        'Local-gov': 4,
        'State-gov': 5, 
        'Without-pay': 6,
        'Never-worked': 7,
        '?': 8
    }).astype(int)

testingDataFrame['education'] = testingDataFrame['education'].map(
    {
        'Bachelors': 0, 
        'Some-college': 1, 
        '11th': 2, 
        'HS-grad': 3, 
        'Prof-school': 4, 
        'Assoc-acdm': 5, 
        'Assoc-voc': 6, 
        '9th': 7,
        '7th-8th': 8, 
        '12th': 9, 
        'Masters': 10, 
        '1st-4th': 11,
        '10th': 12,
        'Doctorate': 13,
        '5th-6th': 14,
        'Preschool': 15
    }).astype(int)

testingDataFrame['occupation'] = testingDataFrame['occupation'].map(
    { 
        'Tech-support': 1, 
        'Craft-repair': 2, 
        'Other-service': 3, 
        'Sales': 4, 
        'Exec-managerial': 5,
        'Prof-specialty': 6, 
        'Handlers-cleaners': 7,
        'Machine-op-inspct': 8,
        'Adm-clerical': 9,
        'Farming-fishing': 10, 
        'Transport-moving': 11, 
        'Priv-house-serv': 12, 
        'Protective-serv': 13,
        'Armed-Forces': 14,
        '?': 15
    }).astype(int)

testingDataFrame['relationship'] = testingDataFrame['relationship'].map(
    {
        'Wife': 0, 
        'Own-child': 1, 
        'Husband': 2, 
        'Not-in-family': 3,
        'Other-relative': 4,
        'Unmarried': 5
    }).astype(int)

#Start getting data ready for learning regression

trainingData_x = pd.DataFrame(
    np.c_[trainingDataFrame['relationship'], trainingDataFrame['education'], trainingDataFrame['fnlwgt'], 
    trainingDataFrame['race'], trainingDataFrame['occupation'], trainingDataFrame['sex'],
    trainingDataFrame['marital.status'], trainingDataFrame['workclass'], trainingDataFrame['age'], 
    trainingDataFrame['capital.gain'], trainingDataFrame['capital.loss'], trainingDataFrame['hours.per.week']], 
    columns = ['relationship','education','fnlwgt', 'race','occupation','gender','marital.status','workclass', 'age', 'capital.gain', 'capital.loss', 'hours.per.week'])

trainingData_y = pd.DataFrame(trainingDataFrame['income>50K'])

testingData = pd.DataFrame(
    np.c_[testingDataFrame['relationship'], testingDataFrame['education'], testingDataFrame['fnlwgt'], 
    testingDataFrame['race'], testingDataFrame['occupation'], testingDataFrame['sex'],
    testingDataFrame['marital.status'], testingDataFrame['workclass'], testingDataFrame['age'], 
    testingDataFrame['capital.gain'], testingDataFrame['capital.loss'], testingDataFrame['hours.per.week']], 
    columns = ['relationship','education','fnlwgt', 'race','occupation','gender','marital.status','workclass', 'age', 'capital.gain', 'capital.loss', 'hours.per.week'])

logisticReg = LogisticRegression(max_iter=500)

logisticReg.fit(trainingData_x, trainingData_y.values.ravel())

LogisticRegressionPredictions = logisticReg.predict_proba(testingData)[:,1]

print(LogisticRegressionPredictions)

with open(r"LogisticRegressionOutput.csv", 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Prediction'])
    for i in range(len(LogisticRegressionPredictions)):
        writer.writerow([i+1,LogisticRegressionPredictions[i]])


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(trainingData_x, trainingData_y.values.ravel())

randomForestPredictions = clf.predict_proba(testingData)[:,1]

print(randomForestPredictions)

with open(r"RandomForestOutput.csv", 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Prediction'])
    for i in range(len(randomForestPredictions)):
        writer.writerow([i+1,randomForestPredictions[i]])
