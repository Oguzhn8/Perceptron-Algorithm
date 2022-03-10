import random
import numpy as np
from copy import deepcopy
import pandas as pd

original = []

# We have defined a data set to keep the results we will obtain by changing the k (learning rate) value.
df_k = pd.DataFrame(columns=["k", "w", "success", "w count"])


# We have defined a data set to keep the results we will obtain by changing the w(initial weight) value.
df_w = pd.DataFrame(columns=["k", "w", "success", "w count"])


# We have defined a data set where we will keep the results we will obtain by mixing our model training values.
df_s = pd.DataFrame(columns=["k", "w", "success", "w count"])

for j in range(50):                                 
# Let's create a dataset of 50 points
    randomlist = []
    for i in range(0, 6):
        n = random.uniform(-0.5, 0.5)
        randomlist.append(n)

    if randomlist[3] >= 0:
        original.append([randomlist, -1])
    else:
        original.append([randomlist, 1])

dots = deepcopy(original)                           # Let's save 50 points
random.shuffle(dots)                                #shuffle the dots

dots_train = dots[:30]                              #assign the first 30 points to the training set
dots_test = dots[30:]                               #assign the last 20 points to the test set


success_count = 0
step = 1

for w_x in range(20):                               #train our training set by changing the learning rate

    k_learn = 0 + w_x*0.005
    success_count = 0
    w_list = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    for i in range(100):
        dots = deepcopy(original)
        dots_train = dots[:30]
        if success_count >= len(dots_train):
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
                w = w_list[-1] + 0.5 * k_learn * (target - calculated_target) * dot  # let's update our weights
                w_list.append(w)

    test_success = 0                                #Let's test our dataset with our testset
    dots = deepcopy(original)
    dots_test = dots[30:]
    for j in range(len(dots_test)):
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
    df_k.loc[step] = [k_learn, w_list[0][0], test_success / len(dots_test), len(w_list)]    # save the results

    step += 1

step = 1
for w_x in range(20):                           #train our dataset by changing the initial conditions of the weights

    k_learn = 0.1
    success_count = 0
    w_1 = -0.1
    w_inc = w_x * 0.01
    w = [w_1 + w_inc, w_1 + w_inc, w_1 + w_inc, w_1 + w_inc, w_1 + w_inc, w_1 + w_inc, w_1 + w_inc]
    w_list = [w]
    for i in range(100):
        dots = deepcopy(original)
        dots_train = dots[:30]
        if success_count >= len(dots_train):
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
                w = w_list[-1] + 0.5 * k_learn * (target - calculated_target) * dot # update the weights
                w_list.append(w)

    test_success = 0
    dots = deepcopy(original)
    dots_test = dots[30:]
    for j in range(len(dots_test)):     #Let's test with our testset
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
    df_w.loc[step] = [k_learn, w_list[0][0], test_success / len(dots_test), len(w_list)]    # save the results

    step += 1

step = 1
for w_x in range(20):           #train our training set by mixing our dataset every cycle

    dots = deepcopy(original)
    dots_training = dots[:30]
    random.shuffle(dots_training)           

    k_learn = 0.1
    success_count = 0
    w_list = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    for i in range(100):
        dots_train = deepcopy(dots_training)
        if success_count >= len(dots_train):
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
                w = w_list[-1] + 0.5 * k_learn * (target - calculated_target) * dot # update the weights
                w_list.append(w)

    test_success = 0                        #testing
    dots = deepcopy(original)
    dots_test = dots[30:]
    for j in range(len(dots_test)):
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
    df_s.loc[step] = [k_learn, w_list[0][0], test_success / len(dots_test), len(w_list)] # save the results

    step += 1
#print the saved results
print(df_k) 
print(df_w)
print(df_s)
