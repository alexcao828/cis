import numpy as np
import tensorflow as tf
from policy_net import policy_net
from larm import larm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

x_train = np.load('x_train.npy')
x_validation = np.load('x_validation.npy')
x_test = np.load('x_test.npy')

y_train = np.load('y_train.npy')
y_validation = np.load('y_validation.npy')
y_test = np.load('y_test.npy')

MU = 1e-3
batch_size = 128
num_epochs = 10000
num_classes = 2

num_training = len(y_train)
num_mbs_training = int(num_training/batch_size)
num_validation = len(y_validation)
num_mbs_validation = int(num_validation/batch_size)
num_test = len(y_test)
num_mbs_test = int(num_test/batch_size)

total_words = np.amax(x_train)+1
num_time_steps = len(x_train[0])

policy = policy_net(total_words, num_time_steps)
LARM = larm(policy, num_time_steps, MU)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_acc_epoch = []
    train_r_epoch = []
    train_racc_epoch = []
    train_T_epoch = []

    val_acc_epoch = []
    val_r_epoch = []
    val_racc_epoch = []
    val_T_epoch = []
    allValActProbs = np.zeros((num_epochs, num_mbs_validation * batch_size, num_time_steps))
    allValClassProbs = np.zeros((num_epochs, num_mbs_validation * batch_size, num_time_steps))

    test_acc_epoch = []
    test_r_epoch = []
    test_racc_epoch = []
    test_T_epoch = []

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
            x_mb = np.take(x_train, here, axis=0)
            ys = np.copy(y_train[here])
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

        val_y = []
        val_guess = []
        val_r = 0.0
        val_racc = 0.0
        val_T = 0.0
        actProbsEpoch = np.zeros((num_mbs_validation * batch_size, num_time_steps))
        classProbsEpoch = np.zeros((num_mbs_validation * batch_size, num_time_steps))
        for v in range(num_mbs_validation):
            x_mb = x_validation[v * batch_size:(v + 1) * batch_size]
            ys = np.copy(y_validation[v * batch_size:(v + 1) * batch_size])
            y_mb = np.eye(num_classes)[ys]

            Ts, rs_acc, guess, actProbsOut, classProbsOut = LARM.eval(x_mb, y_mb)

            val_y.extend(np.argmax(y_mb, 1))
            val_guess.extend(guess)
            val_r += np.mean(rs_acc - MU*Ts) / num_mbs_validation
            val_racc += np.mean(rs_acc) / num_mbs_validation
            val_T += np.mean(Ts) / num_mbs_validation
            actProbsEpoch[v * batch_size:(v + 1) * batch_size, :] = actProbsOut
            classProbsEpoch[v * batch_size:(v + 1) * batch_size, :] = classProbsOut

        val_acc_epoch.append(accuracy_score(val_y, val_guess))
        val_r_epoch.append(val_r)
        val_racc_epoch.append(val_racc)
        val_T_epoch.append(val_T)
        allValActProbs[epoch, :, :] = actProbsEpoch
        allValClassProbs[epoch, :, :] = classProbsEpoch

        test_y = []
        test_guess = []
        test_r = 0.0
        test_racc = 0.0
        test_T = 0.0
        for t in range(num_mbs_test):
            x_mb = x_test[t * batch_size:(t + 1) * batch_size]
            ys = np.copy(y_test[t * batch_size:(t + 1) * batch_size])
            y_mb = np.eye(num_classes)[ys]

            Ts, rs_acc, guess, _, _ = LARM.eval(x_mb, y_mb)

            test_y.extend(np.argmax(y_mb, 1))
            test_guess.extend(guess)
            test_r += np.mean(rs_acc - MU*Ts) / num_mbs_test
            test_racc += np.mean(rs_acc) / num_mbs_test
            test_T += np.mean(Ts) / num_mbs_test

        test_acc_epoch.append(accuracy_score(test_y, test_guess))
        test_r_epoch.append(test_r)
        test_racc_epoch.append(test_racc)
        test_T_epoch.append(test_T)

        plt.figure(figsize=(12, 8))
        plt.plot(train_acc_epoch, 'bs-', label='train')
        plt.plot(val_acc_epoch, 'gs-', label='val')
        plt.plot(test_acc_epoch, 'rs-', label='test')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig('acc.png')
        plt.close()

        np.save('val_acc.npy', np.asarray(val_acc_epoch))
        np.save('test_acc.npy', np.asarray(test_acc_epoch))

        plt.figure(figsize=(12, 8))
        plt.plot(train_r_epoch, 'bs-', label='train')
        plt.plot(val_r_epoch, 'gs-', label='val')
        plt.plot(test_r_epoch, 'rs-', label='test')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('r')
        plt.savefig('r.png')
        plt.close()

        np.save('val_r.npy', np.asarray(val_r_epoch))
        np.save('test_r.npy', np.asarray(test_r_epoch))

        plt.figure(figsize=(12, 8))
        plt.plot(train_racc_epoch, 'bs-', label='train')
        plt.plot(val_racc_epoch, 'gs-', label='val')
        plt.plot(test_racc_epoch, 'rs-', label='test')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('r acc')
        plt.savefig('r_acc.png')
        plt.close()

        np.save('val_racc.npy', np.asarray(val_racc_epoch))
        np.save('test_racc.npy', np.asarray(test_racc_epoch))

        plt.figure(figsize=(12, 8))
        plt.plot(train_T_epoch, 'bs-', label='train')
        plt.plot(val_T_epoch, 'gs-', label='val')
        plt.plot(test_T_epoch, 'rs-', label='test')
        plt.legend(loc='upper right', prop={'size': 20})
        plt.xlabel('epoch')
        plt.ylabel('T')
        plt.savefig('T.png')
        plt.close()

        np.save('val_T.npy', np.asarray(val_T_epoch))
        np.save('test_T.npy', np.asarray(test_T_epoch))

        np.save('allValActProbs' + str(epoch) + '.npy', allValActProbs[epoch])
        np.save('allValClassProbs' + str(epoch) + '.npy', allValClassProbs[epoch])
