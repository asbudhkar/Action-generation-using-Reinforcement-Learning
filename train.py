import numpy as np # linear algebra
import pandas as pd # data processing,  CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import pickle
import cv2
import time
import os
import sklearn
# Any results you write to the current directory are saved as output.

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.utils import shuffle

#used to generate input and output parameters from a csv file
def generator(filename):
            file1 = np.loadtxt(open(filename,  "rb"),  delimiter = ",",  skiprows = 0)
            print(file1)
            #delete first two rows as they dont have acceleration
            file1  =  np.delete(file1,  0,  axis = 0)
            file1  =  np.delete(file1,  0,  axis = 0)      
            arr = np.empty((0, 150), float) #input(v, d): 25*3*2
            arr1 = np.empty((0, 75), float) #output(a): 25*3
            for mat in file1:
                all_val1 = []
                all_val = []
                for i in range(0, 225, 9):
                    v  =  np.array([mat[i], mat[i+1], mat[i+2], mat[i+3], mat[i+4], mat[i+5]]) #populate the input
                    v1  =  np.array([mat[i+6], mat[i+7], mat[i+8]])                         #populate the output
                    all_val.append(v)
                    all_val1.append(v1)
                all_val = np.reshape(all_val, (1, 150))
                all_val1 = np.reshape(all_val1, (1, 75))

                arr  = np.append(arr, all_val, axis = 0)
                arr1 = np.append(arr1, all_val1, axis = 0)
            print(arr.shape)
            print(arr1.shape)
            X_train = arr
            y_train = arr1
            return X_train, y_train

print("------------------------------")

# making the model
X  =  tf.placeholder("float",  [None,  150])                        # input
Y  =  tf.placeholder("float",  [None,  75])                         # true output
W  =  tf.Variable(tf.random_uniform([150, 100], -1, 1))             # Hidden Layer 1  
b  =  tf.Variable(tf.random_uniform([100], -1, 1))      
W1 =  tf.Variable(tf.random_uniform([100, 75], -1, 1))              # Hidden Layer 2
b1 =  tf.Variable(tf.random_uniform([75], -1, 1))      
y  =  tf.nn.relu(tf.matmul(X, W) + b)
y1 =  (tf.matmul(y, W1) + b1)                          # predicted output

learning_rate  =  tf.placeholder("float",  [])
frame_number = tf.placeholder("float", [])

cross_entropy  =  tf.reduce_mean(tf.squared_difference(y1,  Y))     #   Calculating loss by MSE
cross_entropy = cross_entropy*pow(0.95,60-frame_number)
correct_prediction  =  y1
opt  =  tf.train.AdamOptimizer(0.01,  0.9)                          # Setting up the optimizer
train_op  =  opt.minimize(cross_entropy)
init  =  tf.initialize_all_variables()
sess  =  tf.Session()
saver  =  tf.train.Saver()
f=open("output.csv",'ab')
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # Training Phase   
    for j in range (1):
        for k in os.listdir('./output2'):
            X_train, y_train = generator('./output2/'+os.fsdecode(k))

            for i in range (0,  60):
                feed_dict = {
                            X: X_train[i-1:i], 
                            Y: y_train[i-1:i], 
                            learning_rate: 0.1, 
                            frame_number:i
                    }
                cor, loss, _ = sess.run([correct_prediction, cross_entropy, train_op],  feed_dict = feed_dict)

                print(loss)
                print(cor)
                if i % 512  ==  0:
                   print("training on image #%d" % i)

    print("Epoch ", j)
    
    file1  =  np.loadtxt(open("./output2/file_3.csv",  "rb"),  delimiter = ",",  skiprows = 0)
    file1  =  np.delete(file1,  0,  axis = 0)
    file1  =  np.delete(file1,  0,  axis = 0)

    Xtrain = X_train[1:2]
            
    ytrain = y_train[1:2]

    #   Testing phase:- Generating next 60 frames
    for j in range(1,  60):
            if j == 1:
                mat = file1[j:j+1]
            else:
                mat = mat1       
            feed_dict = {
                        X:Xtrain, 
                        Y:ytrain, 
                        learning_rate:0.01, 
                        frame_number:j
                }

            cor, loss, _ = sess.run([correct_prediction, cross_entropy, train_op],  feed_dict = feed_dict)

            print(cor)
            mat1 = []
            all_val = []
            v1_val = []
            v_val = []
            k = 0

            for i in range(0, 225, 9):
                a =  np.array([cor[0][k], cor[0][k+1], cor[0][k+2]])
                v  =  np.array([mat[0][i+3]+cor[0][k], mat[0][i+4]+cor[0][k+1], mat[0][i+5]+cor[0][k+2]])
                v1  =  np.array([mat[0][i]+mat[0][i]+cor[0][k], mat[0][i+1]+mat[0][i+1]+cor[0][k+1], mat[0][i+2]+mat[0][i+2]+cor[0][k+2]])
                k = k+3
                v1_val.extend(v1)
                v_val.extend(v)
                all_val.extend(v1)
                all_val.extend(v)
                mat1.extend(v1)
                mat1.extend(v)
                mat1.extend(a)

            mat1 = np.reshape(mat1, (1, 225))
            all_val  =  np.reshape(all_val, (1, 150))
            v1_val  =  np.reshape(v1_val, (1, 75))   
            v_val  =  np.reshape(v_val, (1, 75))         
            Xtrain  =  all_val
            ytrain  =  y_train[j-1:j]
    f=open('out.csv','ab')
    np.savetxt(f,v1_val,delimiter = "\n")
            
sess.close()