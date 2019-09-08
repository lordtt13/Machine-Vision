# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 23:46:33 2019

@author: tanma
"""

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model_file('model.h5')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()

from keras.models import load_model
keras_model = load_model('model.h5')
model = load_model('model.h5')
model.save_weights(tflite_quant_model)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model.predict(X_test,batch_size = 10,verbose = 1)
keras_model.predict(X_test,batch_size = 10,verbose = 1)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,model.predict(X_test)))
print(roc_auc_score(y_test,keras_model.predict(X_test)))

model.save('tflite_model.h5')