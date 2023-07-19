import numpy as np
import tensorflow as tf
from policy_net import policy_net
from imitate import imitate
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random

train_features = np.load('train_features.npy')
val_features = np.load('val_features.npy')
test_features = np.load('test_features.npy')

train_profit = np.load('train_profit.npy')
val_close = np.load('val_close.npy')
test_close = np.load('test_close.npy')

MU = 1e-3
batch_size = 128
num_epochs = 10000
num_classes = 2

num_training = len(train_profit)
num_mbs_training = int(num_training/batch_size)
num_validation = np.shape(val_close)[0]
num_test = np.shape(test_close)[0]

num_time_steps = np.shape(train_features)[1]
num_features = np.shape(train_features)[2]

policy = policy_net(num_time_steps, num_features)
IMITATE = imitate(policy, num_time_steps, num_features, MU)

flag = 0
for stock in range(np.shape(val_features)[0]):
    start = 0
    ending = start + num_time_steps
    while ending <= np.shape(val_features)[1]:
        if flag == 0:
            valFeaturesReshaped = val_features[stock:stock+1, start:ending, :]
            flag = 1
        elif flag == 1:
            temp = val_features[stock:stock+1, start:ending, :]
            valFeaturesReshaped = np.concatenate((valFeaturesReshaped, temp), axis=0)
        start = int(start + num_time_steps/2)
        ending = start + num_time_steps

flag = 0
for stock in range(np.shape(val_close)[0]):
    start = 0
    ending = start + num_time_steps
    while ending <= np.shape(val_close)[1]:
        if flag == 0:
            valCloseReshaped = val_close[stock:stock+1, start:ending]
            flag = 1
        elif flag == 1:
            temp = val_close[stock:stock+1, start:ending]
            valCloseReshaped = np.concatenate((valCloseReshaped, temp), axis=0)
        start = int(start + num_time_steps/2)
        ending = start + num_time_steps

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_acc_epoch = []
    train_T_epoch = []

    val_acc_epoch = []
    val_T_epoch = []
    val_profit_epoch = []

    test_acc_epoch = []
    test_T_epoch = []
    test_profit_epoch = []

    indices = np.arange(num_training)
    for epoch in range(num_epochs):
        np.random.shuffle(indices)

        train_acc = 0.0
        train_T = 0.0
        for t in range(num_mbs_training):
            here = indices[t * batch_size:(t + 1) * batch_size]
            x_mb = np.take(train_features, here, axis=0)
            ys = np.copy(train_profit[here])
            y_mb = np.eye(num_classes)[ys]
            guesses, Ts = IMITATE.update(x_mb, y_mb)
            train_acc += accuracy_score(ys, guesses) / num_mbs_training
            train_T += np.mean(Ts) / num_mbs_training
        train_acc_epoch.append(train_acc)
        train_T_epoch.append(train_T)

        val_acc_temp = []
        val_T_temp = []
        val_profit_temp = []
        for v in range(num_validation):
            x_mb = val_features[v]
            ys = val_close[v]
            Ts, corrects, profits = IMITATE.evalYear(x_mb, ys)
            val_acc_temp.extend(corrects)
            val_T_temp.extend(Ts)
            val_profit_temp.extend(profits)
        val_acc_epoch.append(np.mean(val_acc_temp))
        val_T_epoch.append(np.mean(val_T_temp))
        val_profit_epoch.append(np.mean(val_profit_temp))

        numRounds = int(np.shape(valFeaturesReshaped)[0] / batch_size)
        allActProbs = np.zeros((numRounds*batch_size, num_time_steps))
        for t in range(numRounds):
            x_mb = valFeaturesReshaped[t * batch_size:(t + 1) * batch_size]
            actProbs = IMITATE.getActProbs(x_mb)
            allActProbs[t * batch_size:(t + 1) * batch_size, :] = actProbs
        np.save('actProbs'+str(epoch)+'.npy', allActProbs)





        test_acc_temp = []
        test_T_temp = []
        test_profit_temp = []
        for t in range(num_test):
            x_mb = test_features[t]
            ys = test_close[t]
            Ts, corrects, profits = IMITATE.evalYear(x_mb, ys)
            test_acc_temp.extend(corrects)
            test_T_temp.extend(Ts)
            test_profit_temp.extend(profits)
        test_acc_epoch.append(np.mean(test_acc_temp))
        test_T_epoch.append(np.mean(test_T_temp))
        test_profit_epoch.append(np.mean(test_profit_temp))

        plt.figure(figsize=(12, 8))
        plt.plot(train_acc_epoch, 'bs-', label='train ideal acc')
        plt.plot(val_acc_epoch, 'gs-', label='val')
        plt.plot(test_acc_epoch, 'rs-', label='test')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.savefig('acc.png')
        plt.close()
        np.save('val_acc.npy', np.asarray(val_acc_epoch))
        np.save('test_acc.npy', np.asarray(test_acc_epoch))

        plt.figure(figsize=(12, 8))
        plt.plot(train_T_epoch, 'bs-', label='train ideal T')
        plt.plot(val_T_epoch, 'gs-', label='val')
        plt.plot(test_T_epoch, 'rs-', label='test')
        plt.legend(loc='upper right', prop={'size': 20})
        plt.savefig('T.png')
        plt.close()
        np.save('val_T.npy', np.asarray(val_T_epoch))
        np.save('test_T.npy', np.asarray(test_T_epoch))

        plt.figure(figsize=(12, 8))
        plt.plot(val_profit_epoch, 'gs-', label='val')
        plt.plot(test_profit_epoch, 'rs-', label='test')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.savefig('profit.png')
        plt.close()
        np.save('val_profit.npy', np.asarray(val_profit_epoch))
        np.save('test_profit.npy', np.asarray(test_profit_epoch))
