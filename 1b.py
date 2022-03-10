import random
import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt

randomlist = []
original = []


for j in range(50):                     
    randomlist = []
    for i in range(0, 6):
        n = random.uniform(-10, 10)
        randomlist.append(n)
    
    if randomlist[3] > -5 and randomlist[3] < 5:  #classify our dataset so that it can be separated by at least two lines
        original.append([randomlist, -1])
    else:
        original.append([randomlist, 1])
        
dots = deepcopy(original)   # Let's save 50 points
random.shuffle(dots)   #shuffle the dataset


dots_train = dots[:30]      # Let's import 30 data into our training set
dots_test = dots[30:]       #  import 20 data into our test set



    
k_learn = 0.1           #learning rate 
success_count = 0
w_list = [[1, 1, 1, 1, 1, 1, 1]]    #first weights
#train the model
for i in range(100):                
        dots = deepcopy(original)
        dots_train = dots[:30]
        if success_count >= len(dots_train):    #durdurma krteri
            break
        for j in range(len(dots_train)):
            dot = dots_train[j][0]
            target = dots_train[j][1]
            dot.append(1)
            dot = np.asmatrix(dot)
            dot_T = np.transpose(dot)

            result = np.matmul(w_list[-1], dot_T)
            calculated_target = 0
            if result >= 0:
                calculated_target = 1
            else:
                calculated_target = -1

            if calculated_target == target:
                success_count += 1
            else:
                success_count = 0
                w = w_list[-1] + 0.5 * k_learn * (target - calculated_target) * dot#ağırlık güncellemesi
                w_list.append(w)

test_success = 0
dots = deepcopy(original)
dots_test = dots[30:]
for j in range(len(dots_test)):     #test the model
        dot = dots_test[j][0]
        target = dots_test[j][1]
        dot.append(1)
        dot = np.asmatrix(dot)
        dot_T = np.transpose(dot)

        result = np.matmul(w_list[-1], dot_T)
        calculated_target = 0
        if result >= 0:
            calculated_target = 1
        else:
            calculated_target = -1

        if calculated_target == target:
            test_success += 1
    
success_rate = test_success/len(dots_test)  #success rate
w_count = len(w_list)                       #step count for obtaining right weights

print("Success:",success_rate)
print("W_count:",w_count)