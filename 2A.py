import numpy as np
import random
from copy import deepcopy
#linearly unseperable dataset
data_set = [[[0, -1], 1],
            [[0, 0], 1],
            [[0, 1], 1],
            [[1, -1], 1],
            [[1, 0], 1],
            [[1, 1], 1],
            [[-1, -1], 1],
            [[-1, 0], 1],
            [[-1, 1], 1],
            [[-3, 3], -1],
            [[-3, 1], -1],
            [[-3, 0], -1],
            [[-3, -1], -1],
            [[-3, -3], -1],
            [[-1, 3], -1],
            [[-1, -3], -1],
            [[0, 3], -1],
            [[0, -3], -1],
            [[1, 3], -1],
            [[1, -3], -1],
            [[3, 3], -1],
            [[3, 1], -1],
            [[3, 0], -1],
            [[3, -1], -1],
            [[3, -3], -1],
            [[-2, 3], -1],
            [[-3, 2], -1],
            [[-3, -2], -1],
            [[-2, -3], -1],
            [[2, 3], -1],
            [[3, 2], -1],
            [[3, -2], -1],
            [[2, -3], -1]]

random.shuffle(data_set)
dots = deepcopy(data_set)
#let's determine our training and test set
dots_train = dots[:19]
dots_test = dots[19:]

#determine our middleware units by using the equations we have obtained.
success_count = 0
step = 1
v_1 = [[-1.67, -1], 3]
v_2 = [[1.67, -1], 3]
v_3 = [[0, 1], 1.5]

k_learn = 1
w_list = [[1, 1, 1]]
#eğitime başlayalım
for i in range(1000):
    if success_count >= len(dots_train):#stopping criteria
        break
    for j in range(len(dots_train)):
        dot = dots_train[j][0]
        target = dots_train[j][1]
        dot = np.asmatrix(dot)
        dot_T = np.transpose(dot)
        v = [0, 0, 0]
        v = np.array(v)
        if np.matmul(v_1[0], dot_T) >= 3:   #middleware unit condition 1
            v[0] = -1
        else:
            v[0] = 1

        if np.matmul(v_2[0], dot_T) >= 3:   #middleware unit condition 2
            v[1] = -1
        else:
            v[1] = 1

        if np.matmul(v_3[0], dot_T) >= 1.5: #middleware unit condition 3
            v[2] = -1
        else:
            v[2] = 1

        result = np.matmul(w_list[-1], np.transpose(v))
        calculated_target = 0
        if result >= 3:
            calculated_target = 1
        else:
            calculated_target = -1

        if calculated_target == target:
            success_count += 1
        else:
            success_count = 0
            w = w_list[-1] + 0.5 * k_learn * (target - calculated_target) * v
            w_list.append(w)        #update weights
    step = step + 1

    if step == 1000:      
        print("Eğitim tamamlanamadı")

success_count = 0
for j in range(len(dots_test)):     #testing
    dot = dots_test[j][0]
    target = dots_test[j][1]
    dot = np.asmatrix(dot)
    dot_T = np.transpose(dot)
    v = [0, 0, 0]
    v = np.array(v)
    if np.matmul(v_1[0], dot_T) >= 3:   
        v[0] = -1
    else:
        v[0] = 1

    if np.matmul(v_2[0], dot_T) >= 3:   
        v[1] = -1
    else:
        v[1] = 1

    if np.matmul(v_3[0], dot_T) >= 1.5: 
        v[2] = -1
    else:
        v[2] = 1

    result = np.matmul(w_list[-1], np.transpose(v))
    calculated_target = 0
    if result >= 3:
        calculated_target = 1
    else:
        calculated_target = -1

    if calculated_target == target:
        success_count += 1
    else:
        success_count = 0

print(success_count/len(dots_test)) #success rate
