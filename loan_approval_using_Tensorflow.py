# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:05:25 2018

@author: Rahul9.Mishra
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2

data = pd.read_csv(r'train.csv')

'''
Data cleaning start here

1) Remove unwanted columns and convert chracters into numbers
2) Fill nan Values
3) Remove noise
4) Feature engineering if necessary
5) Dimentionality reduction if necessary
'''

#remove the unwated columns, here we do not need 'Loan_ID' column since it dosent add any value to the data

cols = ['Gender', 'Married', 'Dependents', 'LoanAmount', 'Credit_History', 'Property_Area', 'Loan_Status']

data = data[cols]

#fill the NaN values while we convert categorical data to numeric

data.Gender = data.Gender.map({'Male' : 1, 'Female' : 0, np.nan : -1})
data.Married = data.Married.map({'No' : 0, 'Yes' : 1, np.nan : -1})
data.Dependents = data.Dependents.map({'0' : 0, '1' : 1, '2' : 2, '3+' : 3, np.nan : -1})
data.Credit_History = data.Credit_History.map({1. : 1, 0. : 0, np.nan : -1})
data.Property_Area = data.Property_Area.map({'Urban' : 1, 'Rural' : -1, 'Semiurban' : 0})
data.Loan_Status = data.Loan_Status.map({'Y' : 1, 'N' : 0})

#Remove noise from data
#we will remove the rows where Loan amount isnt mentioned since that can drastically imapct the model predicting

data = data[data.LoanAmount.isnull() == False]

#split the data into training and validation 

feature_data = data[['Gender', 'Married', 'Dependents', 'LoanAmount', 'Credit_History', 'Property_Area']]
label_data = data['Loan_Status']

#Dimentionality reduction
imp = SelectPercentile(percentile=50).fit(feature_data, label_data).get_support()

#since we found only these 3 columns have great impact on the outcome
feature_data =  feature_data[['Married', 'Credit_History', 'Property_Area']]
x_train, x_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=.3)


'''
Tensorflow implementation starts here

1) Create input function that will feed the data to the network
2) Create the neural network architecture
3) Train the model
4) Test the model accuracy

'''

#create input function
feature_column = [tf.contrib.layers.real_valued_column(x) for x in x_train.columns]

def input_fn(df, target):
    
    features = {x: tf.constant(df[x].values, shape= [df.shape[0]]) for x in df.columns}
    labels = tf.constant(target.values, shape=[target.shape[0]])

    return features, labels

#create neural network architecture
    
classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_column, hidden_units=[10,5,10], n_classes=2)

#train the model
classifier = classifier.fit(input_fn = lambda :input_fn(x_train, y_train), steps=1000)

#test the model accuracy
ev = classifier.evaluate(input_fn = lambda :input_fn(x_test, y_test), steps=1)

print('The accuracy is \n', ev)

'''
Accuracy of the model was 84%

'''













