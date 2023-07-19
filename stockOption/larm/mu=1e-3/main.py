import numpy as np
import tensorflow as tf
from policy_net import policy_net
from larm import larm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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
num_trials = 100

num_training = np.shape(train_features)[0]
num_mbs_training = int(num_training/batch_size)
num_validation = np.shape(val_features)[0]
num_test = np.shape(test_features)[0]

num_time_steps = np.shape(train_features)[1]
num_features = np.shape(train_features)[2]

policy = policy_net(num_time_steps, num_features)
LARM = larm(policy, num_time_steps, MU)

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
    train_r_epoch = []
    train_racc_epoch = []
    train_T_epoch = []

    val_T_mean = []
    val_T_se = []
    val_acc_mean = []
    val_acc_se = []
    val_profit_mean = []
    val_profit_se = []

    test_T_mean = []
    test_T_se = []
    test_acc_mean = []
    test_acc_se = []
    test_profit_mean = []
    test_profit_se = []

    indices = np.arange(num_training)
    for epoch in range(num_epochs):
        np.random.shuffle(indices)

        train_y = []
        train_guess = []
        train_r = 0.0
        train_racc = 0.0
        train_T = 0.0
        for t in range(num_mbs_training):
            here = indices[t * batch_size:(t + 1) * batch_size]
            x_mb = np.take(train_features, here, axis=0)
            ys = np.copy(train_profit[here])
            y_mb = np.eye(num_classes)[ys]

            Ts, rs_acc, guess = LARM.update(x_mb, y_mb)

            train_y.extend(np.argmax(y_mb, 1))
            train_guess.extend(guess)
            train_r += np.mean(rs_acc - MU*Ts) / num_mbs_training
            train_racc += np.mean(rs_acc) / num_mbs_training
            train_T += np.mean(Ts) / num_mbs_training

        train_acc_epoch.append(accuracy_score(train_y, train_guess))
        train_r_epoch.append(train_r)
        train_racc_epoch.append(train_racc)
        train_T_epoch.append(train_T)

        "val"
        val_T_trials = np.zeros(num_trials)
        val_acc_trials = np.zeros(num_trials)
        val_profit_trials = np.zeros(num_trials)
        for trial in range(num_trials):
            T_trial = []
            correct_trial = []
            profit_trial = []
            for v in range(num_validation):
                Ts, corrects, profits = LARM.evalYear(val_features[v], val_close[v])
                T_trial.extend(Ts)
                correct_trial.extend(corrects)
                profit_trial.extend(profits)
            val_T_trials[trial] = np.mean(np.asarray(T_trial))
            val_acc_trials[trial] = np.mean(np.asarray(correct_trial))
            val_profit_trials[trial] = np.mean(np.asarray(profit_trial))
        val_T_mean.append(np.mean(np.asarray(val_T_trials)))
        val_T_se.append(np.std(np.asarray(val_T_trials)) / np.sqrt(num_trials))
        val_acc_mean.append(np.mean(np.asarray(val_acc_trials)))
        val_acc_se.append(np.std(np.asarray(val_acc_trials)) / np.sqrt(num_trials))
        val_profit_mean.append(np.mean(np.asarray(val_profit_trials)))
        val_profit_se.append(np.std(np.asarray(val_profit_trials)) / np.sqrt(num_trials))
        np.save('val_T_mean.npy', val_T_mean)
        np.save('val_T_se.npy', val_T_se)
        np.save('val_acc_mean.npy', val_acc_mean)
        np.save('val_acc_se.npy', val_acc_se)
        np.save('val_profit_mean.npy', val_profit_mean)
        np.save('val_profit_se.npy', val_profit_se)



        numRounds = int(np.shape(valFeaturesReshaped)[0] / batch_size)
        allActProbs = np.zeros((numRounds*batch_size, num_time_steps))
        for t in range(numRounds):
            x_mb = valFeaturesReshaped[t * batch_size:(t + 1) * batch_size]
            actProbs = LARM.getActProbs(x_mb)
            allActProbs[t * batch_size:(t + 1) * batch_size, :] = actProbs
        np.save('actProbs'+str(epoch)+'.npy', allActProbs)




        "test"
        test_T_trials = np.zeros(num_trials)
        test_acc_trials = np.zeros(num_trials)
        test_profit_trials = np.zeros(num_trials)
        for trial in range(num_trials):
            T_trial = []
            correct_trial = []
            profit_trial = []
            for t in range(num_test):
                Ts, corrects, profits = LARM.evalYear(test_features[t], test_close[t])
                T_trial.extend(Ts)
                correct_trial.extend(corrects)
                profit_trial.extend(profits)
            test_T_trials[trial] = np.mean(np.asarray(T_trial))
            test_acc_trials[trial] = np.mean(np.asarray(correct_trial))
            test_profit_trials[trial] = np.mean(np.asarray(profit_trial))
        test_T_mean.append(np.mean(np.asarray(test_T_trials)))
        test_T_se.append(np.std(np.asarray(test_T_trials)) / np.sqrt(num_trials))
        test_acc_mean.append(np.mean(np.asarray(test_acc_trials)))
        test_acc_se.append(np.std(np.asarray(test_acc_trials)) / np.sqrt(num_trials))
        test_profit_mean.append(np.mean(np.asarray(test_profit_trials)))
        test_profit_se.append(np.std(np.asarray(test_profit_trials)) / np.sqrt(num_trials))
        np.save('test_T_mean.npy', test_T_mean)
        np.save('test_T_se.npy', test_T_se)
        np.save('test_acc_mean.npy', test_acc_mean)
        np.save('test_acc_se.npy', test_acc_se)
        np.save('test_profit_mean.npy', test_profit_mean)
        np.save('test_profit_se.npy', test_profit_se)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(val_T_mean, 'b.-', label='val')
        plt.plot(test_T_mean, 'r.-', label='test')
        plt.legend(loc='upper right', prop={'size': 20})
        plt.ylabel('mean T')
        plt.subplot(2, 1, 2)
        plt.plot(val_T_se, 'b.-', label='val')
        plt.plot(test_T_se, 'r.-', label='test')
        plt.legend(loc='upper right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('se T')
        plt.savefig('valTest_T.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(val_acc_mean, 'b.-', label='val')
        plt.plot(test_acc_mean, 'r.-', label='test')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.ylabel('mean acc')
        plt.subplot(2, 1, 2)
        plt.plot(val_acc_se, 'b.-', label='val')
        plt.plot(test_acc_se, 'r.-', label='test')
        plt.legend(loc='upper right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('se acc')
        plt.savefig('valTest_acc.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(val_profit_mean, 'b.-', label='val')
        plt.plot(test_profit_mean, 'r.-', label='test')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.ylabel('mean profit')
        plt.subplot(2, 1, 2)
        plt.plot(val_profit_se, 'b.-', label='val')
        plt.plot(test_profit_se, 'r.-', label='test')
        plt.legend(loc='upper right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('se profit')
        plt.savefig('valTest_profit.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(train_acc_epoch, 'bs-', label='train')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig('train_acc.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(train_r_epoch, 'bs-', label='train')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('r')
        plt.savefig('train_r.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(train_racc_epoch, 'bs-', label='train')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('r acc')
        plt.savefig('train_r_acc.png')
        plt.close()
