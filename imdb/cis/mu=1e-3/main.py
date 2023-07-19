import numpy as np
import tensorflow as tf
from policy_net import policy_net
from imitate import imitate
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random

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
IMITATE = imitate(policy, num_time_steps, MU)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_ce_epoch = []
    train_i_epoch = []
    train_acc_epoch = []
    train_n_epoch = []
    train_racc_epoch = []

    test_ce_epoch = []
    test_i_epoch = []
    test_acc_epoch = []
    test_n_epoch = []
    test_racc_epoch = []

    validation_ce_epoch = []
    validation_i_epoch = []
    validation_acc_epoch = []
    validation_n_epoch = []
    validation_racc_epoch = []
    allValActProbs = np.zeros((num_epochs, num_mbs_validation*batch_size, num_time_steps))
    allValClassProbs = np.zeros((num_epochs, num_mbs_validation * batch_size, num_time_steps))

    indices = np.arange(num_training)
    for epoch in range(num_epochs):
        np.random.shuffle(indices)

        train_ce = 0.0
        train_i = 0.0
        train_acc = 0.0
        train_n = 0.0
        train_racc = 0.0
        for t in range(num_mbs_training):
            here = indices[t * batch_size:(t + 1) * batch_size]
            x_mb = np.take(x_train, here, axis=0)
            ys = np.copy(y_train[here])
            y_mb = np.eye(num_classes)[ys]

            guesses, ce, i, racc, Ts = IMITATE.update(x_mb, y_mb)

            train_ce += ce / num_mbs_training
            train_i += i / num_mbs_training
            train_acc += accuracy_score(ys, guesses) / num_mbs_training
            train_n += np.mean(Ts) / num_mbs_training
            train_racc += np.mean(racc) / num_mbs_training

        train_ce_epoch.append(train_ce)
        train_i_epoch.append(train_i)
        train_acc_epoch.append(train_acc)
        train_n_epoch.append(train_n)
        train_racc_epoch.append(train_racc)

        validation_ce = 0.0
        validation_i = 0.0
        validation_acc = 0.0
        validation_n = 0.0
        validation_racc = 0.0
        actProbsEpoch = np.zeros((num_mbs_validation*batch_size, num_time_steps))
        classProbsEpoch = np.zeros((num_mbs_validation * batch_size, num_time_steps))
        for v in range(num_mbs_validation):
            x_mb = x_validation[v * batch_size:(v + 1) * batch_size]
            ys = np.copy(y_validation[v * batch_size:(v + 1) * batch_size])
            y_mb = np.eye(num_classes)[ys]

            guesses, ce, i, rs_acc, Ts, actProbsOut, classProbsOut = IMITATE.eval(x_mb, y_mb)

            validation_ce += ce / num_mbs_validation
            validation_i += i / num_mbs_validation
            validation_acc += accuracy_score(ys, guesses) / num_mbs_validation
            validation_n += np.mean(Ts) / num_mbs_validation
            validation_racc += np.mean(rs_acc) / num_mbs_validation
            actProbsEpoch[v * batch_size:(v + 1) * batch_size, :] = actProbsOut
            classProbsEpoch[v * batch_size:(v + 1) * batch_size, :] = classProbsOut

        validation_ce_epoch.append(validation_ce)
        validation_i_epoch.append(validation_i)
        validation_acc_epoch.append(validation_acc)
        validation_n_epoch.append(validation_n)
        validation_racc_epoch.append(validation_racc)
        allValActProbs[epoch, :, :] = actProbsEpoch
        allValClassProbs[epoch, :, :] = classProbsEpoch

        test_ce = 0.0
        test_i = 0.0
        test_acc = 0.0
        test_n = 0.0
        test_racc = 0.0
        for t in range(num_mbs_test):
            x_mb = x_test[t * batch_size:(t + 1) * batch_size]
            ys = np.copy(y_test[t * batch_size:(t + 1) * batch_size])
            y_mb = np.eye(num_classes)[ys]

            guesses, ce, i, rs_acc, Ts, _, _ = IMITATE.eval(x_mb, y_mb)

            test_ce += ce / num_mbs_test
            test_i += i / num_mbs_test
            test_acc += accuracy_score(ys, guesses) / num_mbs_test
            test_n += np.mean(Ts) / num_mbs_test
            test_racc += np.mean(rs_acc) / num_mbs_test

        test_ce_epoch.append(test_ce)
        test_i_epoch.append(test_i)
        test_acc_epoch.append(test_acc)
        test_n_epoch.append(test_n)
        test_racc_epoch.append(test_racc)

        plt.figure(figsize=(12, 8))
        plt.plot(train_ce_epoch, 'bs-', label='train ce')
        plt.plot(validation_ce_epoch, 'gs-', label='validation')
        plt.plot(test_ce_epoch, 'rs-', label='test')
        plt.legend(loc='upper right', prop={'size': 20})
        plt.savefig('ce.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(train_i_epoch, 'bs-', label='train imitation')
        plt.plot(validation_i_epoch, 'gs-', label='validation')
        plt.plot(test_i_epoch, 'rs-', label='test')
        plt.legend(loc='upper right', prop={'size': 20})
        plt.savefig('imitation.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(train_acc_epoch, 'bs-', label='train ideal acc')
        plt.plot(validation_acc_epoch, 'gs-', label='validation')
        plt.plot(test_acc_epoch, 'rs-', label='test')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.savefig('acc.png')
        plt.close()

        np.save('validation_acc.npy', np.asarray(validation_acc_epoch))
        np.save('test_acc.npy', np.asarray(test_acc_epoch))

        plt.figure(figsize=(12, 8))
        plt.plot(train_n_epoch, 'bs-', label='train ideal n')
        plt.plot(validation_n_epoch, 'gs-', label='validation')
        plt.plot(test_n_epoch, 'rs-', label='test')
        plt.legend(loc='upper right', prop={'size': 20})
        plt.savefig('n.png')
        plt.close()

        np.save('validation_n.npy', np.asarray(validation_n_epoch))
        np.save('test_n.npy', np.asarray(test_n_epoch))

        plt.figure(figsize=(12, 8))
        plt.plot(train_racc_epoch, 'bs-', label='train ideal r_acc')
        plt.plot(validation_racc_epoch, 'gs-', label='validation')
        plt.plot(test_racc_epoch, 'rs-', label='test')
        plt.legend(loc='lower right', prop={'size': 20})
        plt.savefig('r_acc.png')
        plt.close()

        np.save('validation_racc.npy', np.asarray(validation_racc_epoch))
        np.save('test_racc.npy', np.asarray(test_racc_epoch))

        np.save('allValActProbs'+str(epoch)+'.npy', allValActProbs[epoch])
        np.save('allValClassProbs'+str(epoch)+'.npy', allValClassProbs[epoch])
