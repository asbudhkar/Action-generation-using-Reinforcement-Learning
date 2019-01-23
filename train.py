import numpy as np # linear algebra
import pandas as pd # data processing
import tensorflow as tf
import time
import os
import sklearn

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.utils import shuffle

#used to generate input and output parameters from a csv file
def generator(filename):
            file1 = np.loadtxt(open(filename,  "rb"),  delimiter = ",",  skiprows = 0)
            #print(file1)
            #delete first two rows as they don't have acceleration
            file1  =  np.delete(file1,  0,  axis = 0)
            file1  =  np.delete(file1,  0,  axis = 0)
            num_lines = sum(1 for line in file1)      
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
            #print(arr.shape)
            #print(arr1.shape)
            X_train = arr
            y_train = arr1
            return X_train, y_train, num_lines

# Model
X  =  tf.placeholder("float",  [None,  150])                        # input
Y  =  tf.placeholder("float",  [None,  75])                         # true output
W  =  tf.Variable(tf.random_uniform([150, 200], -1, 1))             # Hidden Layer 1  
b  =  tf.Variable(tf.random_uniform([200], -1, 1))      
W1 =  tf.Variable(tf.random_uniform([200, 100], -1, 1))             # Hidden Layer 2
b1 =  tf.Variable(tf.random_uniform([100], -1, 1))      
W2 =  tf.Variable(tf.random_uniform([100, 75], -1, 1))              # Hidden Layer 3
b2 =  tf.Variable(tf.random_uniform([75], -1, 1))    

y  =  tf.nn.relu(tf.matmul(X, W) + b)
y1 =  tf.nn.relu(tf.matmul(y,W1)+b1)
y1 =  (tf.matmul(y1, W2) + b2)                          # predicted output 
learning_rate  =  tf.placeholder("float",  [])
frame_number = tf.placeholder("float", [])
total_frames = tf.placeholder("float",[])
cross_entropy  =  tf.reduce_mean(tf.pow(y1- Y,2.0))     # Calculating loss by MSE with return for RL
cross_entropy = cross_entropy*pow(0.95,total_frames-frame_number)
correct_prediction  =  y1
opt  =  tf.train.GradientDescentOptimizer(0.001)                          # Setting up the optimizer
train_op  =  opt.minimize(cross_entropy)
#init  =  tf.initialize_all_variables()
sess  =  tf.Session()
saver  =  tf.train.Saver()
f=open("output.csv",'ab')
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Training Phase   
    for j in range (25):
        for k in os.listdir('./output2'):
            X_train, y_train, num_lines = generator('./output2/'+os.fsdecode(k))

            for i in range (0,  num_lines):
                feed_dict = {
                            X: X_train[i-1:i+5], 
                            Y: y_train[i-1:i+5], 
                            learning_rate: 0.001, 
                            frame_number:i,
                            total_frames:num_lines
                    }
                loss,_ = sess.run([cross_entropy,train_op],  feed_dict = feed_dict)            
    print("Epoch ", j)

    file1  =  np.loadtxt(open("./output2/file_1.csv",  "rb"),  delimiter = ",",  skiprows = 0)
    file1  =  np.delete(file1,  0,  axis = 0)
    file1  =  np.delete(file1,  0,  axis = 0)

    Xtrain = X_train[1:2]
            
    ytrain = y_train[1:2]
    f = open('out.csv','ab')
    #   Testing phase:- Generating next 60 frames
    for j in range(1,  30):
            if j == 1:
                mat = file1[j:j+1]
            else:
                mat = mat1       
            feed_dict = {
                        X:Xtrain, 
                        Y:ytrain, 
                        learning_rate:0.001, 
                        frame_number:j,
                        total_frames:30
                }

            cor= sess.run([correct_prediction],  feed_dict = feed_dict)
            cor=np.reshape(cor,(1,75))
            print(np.shape(cor))
            mat1 = []
            all_val = []
            v1_val = []
            v_val = []
            k = 0

            for i in range(0, 225, 9):
                a =  np.array([cor[0][k], cor[0][k+1], cor[0][k+2]])
                v  =  np.array([mat[0][i+3]+cor[0][k], mat[0][i+4]+cor[0][k+1], mat[0][i+5]+cor[0][k+2]])
                v1  =  np.array([mat[0][i]+mat[0][i+3]+cor[0][k], mat[0][i+1]+mat[0][i+4]+cor[0][k+1], mat[0][i+2]+mat[0][i+5]+cor[0][k+2]])
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
            np.savetxt(f,v1_val,delimiter = ",")   
            Xtrain  =  all_val
            ytrain  =  y_train[j-1:j]
           
            print("Actual")
            print(file1[j:j+1])
            print("Predicted")
            print(v1_val)
            
sess.close()