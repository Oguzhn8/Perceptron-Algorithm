import random
import numpy as np
from copy import deepcopy
import pandas as pd

original = []
# k(öğrenme hızı) değerini değiştirerek elde edeceğimiz sonuçları tutacağımız data seti tanımladık.
df_k = pd.DataFrame(columns=["k", "w", "success", "w count"])

# w(başlangıç ağırlı) değerini değiştirerek elde edeceğimiz sonuçları tutacağımız data seti tanımladık.
df_w = pd.DataFrame(columns=["k", "w", "success", "w count"])

# Model eğitim değerlerimizi karıştırarak elde edeceğimiz sonuçları tutacağımız data seti tanımladık.
df_s = pd.DataFrame(columns=["k", "w", "success", "w count"])

for j in range(50):                                 #50 noktalık bir data kümesi oluşturalım
    randomlist = []
    for i in range(0, 6):
        n = random.uniform(-0.5, 0.5)
        randomlist.append(n)

    if randomlist[3] >= 0:
        original.append([randomlist, -1])
    else:
        original.append([randomlist, 1])

dots = deepcopy(original)                           #50 noktayı saklayalım
random.shuffle(dots)                                #noktaları karıştıralım

dots_train = dots[:30]                              #ilk 30 noktayı eğitim kümesine atayalım
dots_test = dots[30:]                               #son 20 noktayı test kümesine atayalım


success_count = 0
step = 1

for w_x in range(20):                               #öğrenme hızını değiştirerek eğitim kümemimizi eğitelim

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
                w = w_list[-1] + 0.5 * k_learn * (target - calculated_target) * dot  #ağırlıklarımızı güncelleyelim
                w_list.append(w)

    test_success = 0                                #test kümemizle veri setimizi test edelim
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
    df_k.loc[step] = [k_learn, w_list[0][0], test_success / len(dots_test), len(w_list)]    #sonuçları kaydedelim

    step += 1

step = 1
for w_x in range(20):                           #ağırlıkların ilk koşullarını değiştirerek veri kümemizi eğitelim

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
                w = w_list[-1] + 0.5 * k_learn * (target - calculated_target) * dot#ağırlıkları güncelleyelim
                w_list.append(w)

    test_success = 0
    dots = deepcopy(original)
    dots_test = dots[30:]
    for j in range(len(dots_test)):     #test kümemizle test edelim
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
    df_w.loc[step] = [k_learn, w_list[0][0], test_success / len(dots_test), len(w_list)]    #sonuçları kaydedelim

    step += 1

step = 1
for w_x in range(20):           #veri kümemizi her döngüde karıştırarak eğitim kümemizi eğitelim

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
                w = w_list[-1] + 0.5 * k_learn * (target - calculated_target) * dot#ağırlıkları güncelleyelim
                w_list.append(w)

    test_success = 0                        #test edelim
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
    df_s.loc[step] = [k_learn, w_list[0][0], test_success / len(dots_test), len(w_list)] #sonuçları kaydedelim

    step += 1
#kaydedilen sonuçları yazdıralım
print(df_k) 
print(df_w)
print(df_s)
