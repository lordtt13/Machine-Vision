# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:16:31 2019

@author: tanma
"""

import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

input_shape = (train_images.shape[1], train_images.shape[2])

model = keras.Sequential([
keras.layers.Flatten(input_shape=input_shape),
keras.layers.Dense(128, activation=tf.nn.relu),
keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

epochs = 3
model.fit(train_images, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

model.save("model.h5")
converter  = tf.lite.TFLiteConverter.from_keras_model_file("model.h5")
tflite_model = converter.convert()

with open("tflite_model.tflite", "wb") as output_file:
    output_file.write(tflite_model)
    
import numpy as np
import tensorflow as tf
import time
example = np.reshape(train_images[0],(1,train_images.shape[1], train_images.shape[2]))

interpreter = tf.lite.Interpreter(model_path="tflite_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array(example, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

start = time.time()
interpreter.invoke()
end = time.time()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
print(end-start)

model = keras.models.load_model('model.h5')
graph = tf.get_default_graph()
global graph
with graph.as_default():
  print(model.predict(example, verbose = 1))