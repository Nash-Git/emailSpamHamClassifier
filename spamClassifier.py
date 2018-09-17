'''
This program implements one of the classical machine learning problems: email classifier.
The program is able to detect whether an email is a ham or spam
The Multinomial Naive Bayes classifier is used to design the spam detection classifier.
The data set is used from UCI machine learning repository
'''

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


def pre_processing():
    #reading data and adding header (column name) to the data frame
    file_path = 'data/emailDataSet'
    column_names = ['Label', 'Email']
    emails_dataset = pd.read_csv(file_path, sep = '\t', names = column_names)

    #print(emails_dataset.head())
    #print(emails_dataset.shape)
    #print(emails_dataset.Label.value_counts())

    #adding another column to the dataframe with a column header 'Class' which contains 0 for ham and 1 for spam
    emails_dataset['Class'] = emails_dataset.Label.map({'ham': 0, 'spam': 1})

    #print(emails_dataset.shape)

    #X is the feature matrix and y is the class series
    X = emails_dataset.Email
    y = emails_dataset.Class

    #print(X.shape)
    #print(y.shape)
    return X, y


def apply_classifier(X_train_vec, X_test_vec, y_train, y_test):
    clf_NB = MultinomialNB()
    clf_NB.fit(X_train_vec, y_train)

    y_pred = clf_NB.predict(X_test_vec)

    print('Percentage of accuracy: ', round(accuracy_score(y_pred, y_test) * 100, 2), '%')


def predict_unknown():
    pass

#X is the feature matrix and y is the class vector
X, y = pre_processing()

#split the dataset to 75% for training and 25% for testing by default
X_train, X_test, y_train, y_test = tts(X, y)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

#count vectorizer in order to tansform text to numbers
cnt_vec = CountVectorizer()

cnt_vec.fit(X_train)
X_train_vec = cnt_vec.transform(X_train)
X_test_vec = cnt_vec.transform(X_test)

apply_classifier(X_train_vec, X_test_vec, y_train, y_test)

