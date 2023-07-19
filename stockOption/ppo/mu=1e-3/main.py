import numpy as np
import tensorflow as tf
from policy_net import policy_net
from ppo import ppo_train
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random

batch_size = 128
num_actions = 2
num_classes = 2

train_features = np.load('train_features.npy')
train_profit = np.load('train_profit.npy')
num_training_episodes = int(np.shape(train_profit)[0])

num_times_thru_training_set = 100
num_iterations = int(num_training_episodes * num_times_thru_training_set)
M = int(num_training_episodes/batch_size)
num_trials = 100

val_features = np.load('val_features.npy')
val_close = np.load('val_close.npy')
num_val_episodes =  int(np.shape(val_close)[0])

test_features = np.load('test_features.npy')
test_close = np.load('test_close.npy')
num_test_episodes =  int(np.shape(test_close)[0])

num_time_steps = np.shape(train_features)[1]
num_features = np.shape(train_features)[2]

policy = policy_net('policy', num_time_steps, num_features)
old_policy = policy_net('old_policy', num_time_steps, num_features)
ppo = ppo_train(policy, old_policy)

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

    # Set up gradient buffers and set values to 0
    grad_buffer_pe = ppo.get_vars()
    for i, g in enumerate(grad_buffer_pe):
        grad_buffer_pe[i] = g * 0

    # Get possible actions
    action_space = np.arange(num_actions)

    T_ep = []
    y_ep = []
    guess_ep = []
    ce_ep = []
    r_ep = []
    racc_ep = []
    rtime_ep = []

    T_batch = []
    acc_batch = []
    ce_batch = []
    r_batch = []
    racc_batch = []
    rtime_batch = []

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

    indices = np.arange(num_training_episodes)
    np.random.shuffle(indices)
    counter = 0
    epoch = 0

    for ep in range(1, num_iterations + 1):
        i = indices[counter]

        stocks = train_features[i]
        action_probs = policy.get_action_prob(stocks)

        c = 0
        complete = False
        # Run through each episode
        while complete == False:
            temp = action_probs[c]
            action = np.random.choice(action_space, p=temp)
            c += 1
            if action == 1 or c == num_time_steps:
                complete = True

        y = np.zeros(num_classes)
        y[int(train_profit[i])] = 1

        pe_grads, guess, ce, r, racc, rtime = ppo.get_grads(stocks,y, c * np.ones(1), np.asarray([c - 1, 1]))

        for j, g in enumerate(pe_grads):
            grad_buffer_pe[j] =grad_buffer_pe[j]+ g

        T_ep.append(c)
        y_ep.append(train_profit[i])
        guess_ep.append(guess)
        ce_ep.append(ce)
        r_ep.append(r)
        racc_ep.append(racc)
        rtime_ep.append(rtime)

        if ep % num_training_episodes == 0:
            np.random.shuffle(indices)
            counter = 0
        else:
            counter += 1

        if ep % (M*batch_size) == 0:
            "val"
            val_T_trials = np.zeros(num_trials)
            val_acc_trials = np.zeros(num_trials)
            val_profit_trials = np.zeros(num_trials)
            for trial in range(num_trials):
                T_trial = []
                correct_trial = []
                profit_trial = []
                for v in range(num_val_episodes):
                    Ts, corrects, profits = ppo.evalYear(val_features[v], val_close[v])
                    T_trial.extend(Ts)
                    correct_trial.extend(corrects)
                    profit_trial.extend(profits)
                val_T_trials[trial] = np.mean(np.asarray(T_trial))
                val_acc_trials[trial] = np.mean(np.asarray(correct_trial))
                val_profit_trials[trial] = np.mean(np.asarray(profit_trial))
            val_T_mean.append(np.mean(np.asarray(val_T_trials)))
            val_T_se.append(np.std(np.asarray(val_T_trials))/np.sqrt(num_trials))
            val_acc_mean.append(np.mean(np.asarray(val_acc_trials)))
            val_acc_se.append(np.std(np.asarray(val_acc_trials))/np.sqrt(num_trials))
            val_profit_mean.append(np.mean(np.asarray(val_profit_trials)))
            val_profit_se.append(np.std(np.asarray(val_profit_trials))/np.sqrt(num_trials))
            np.save('val_T_mean.npy', val_T_mean)
            np.save('val_T_se.npy', val_T_se)
            np.save('val_acc_mean.npy', val_acc_mean)
            np.save('val_acc_se.npy', val_acc_se)
            np.save('val_profit_mean.npy', val_profit_mean)
            np.save('val_profit_se.npy', val_profit_se)

            numRounds = int(np.shape(valFeaturesReshaped)[0] / batch_size)
            allActProbs = np.zeros((numRounds * batch_size, num_time_steps))
            for t in range(numRounds * batch_size):
                x_mb = valFeaturesReshaped[t]
                actProbs = ppo.getActProbs(x_mb)
                allActProbs[t, :] = actProbs
            np.save('actProbs' + str(epoch) + '.npy', allActProbs)
            epoch += 1

            "test"
            test_T_trials = np.zeros(num_trials)
            test_acc_trials = np.zeros(num_trials)
            test_profit_trials = np.zeros(num_trials)
            for trial in range(num_trials):
                T_trial = []
                correct_trial = []
                profit_trial = []
                for t in range(num_test_episodes):
                    Ts, corrects, profits = ppo.evalYear(test_features[t], test_close[t])
                    T_trial.extend(Ts)
                    correct_trial.extend(corrects)
                    profit_trial.extend(profits)
                test_T_trials[trial] = np.mean(np.asarray(T_trial))
                test_acc_trials[trial] = np.mean(np.asarray(correct_trial))
                test_profit_trials[trial] = np.mean(np.asarray(profit_trial))
            test_T_mean.append(np.mean(np.asarray(test_T_trials)))
            test_T_se.append(np.std(np.asarray(test_T_trials))/np.sqrt(num_trials))
            test_acc_mean.append(np.mean(np.asarray(test_acc_trials)))
            test_acc_se.append(np.std(np.asarray(test_acc_trials))/np.sqrt(num_trials))
            test_profit_mean.append(np.mean(np.asarray(test_profit_trials)))
            test_profit_se.append(np.std(np.asarray(test_profit_trials))/np.sqrt(num_trials))
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
        # Update policy gradients based on batch_size parameter
        if ep % batch_size == 0:
            ppo.assign_policy_parameters()
            ppo.update(grad_buffer_pe)

            # Clear buffer values for next batch
            for i, g in enumerate(grad_buffer_pe):
                grad_buffer_pe[i] = g * 0

            temp = np.asarray(T_ep).astype(float)
            T_batch.append(np.mean(temp))
            T_ep = []

            temp1 = np.asarray(y_ep)
            temp2 = np.asarray(guess_ep)
            acc_batch.append(accuracy_score(temp1, temp2))
            y_ep = []
            guess_ep = []

            temp = np.asarray(ce_ep).astype(float)
            ce_batch.append(np.mean(temp))
            ce_ep = []

            temp = np.asarray(r_ep).astype(float)
            r_batch.append(np.mean(temp))
            r_ep = []

            temp = np.asarray(racc_ep).astype(float)
            racc_batch.append(np.mean(temp))
            racc_ep = []

            temp = np.asarray(rtime_ep).astype(float)
            rtime_batch.append(np.mean(temp))
            rtime_ep = []

            smoothed_T = [np.mean(T_batch[max(0, i - 10):i + 1]) for i in range(len(T_batch))]
            plt.figure(figsize=(12, 8))
            plt.plot(T_batch, 'b.', label='train')
            plt.plot(smoothed_T, 'b', label='train smooth')
            plt.legend(loc='upper right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('T')
            plt.savefig('train_T.png')
            plt.close()

            smoothed_acc = [np.mean(acc_batch[max(0, i - 10):i + 1]) for i in range(len(acc_batch))]
            plt.figure(figsize=(12, 8))
            plt.plot(acc_batch, 'b.', label='train')
            plt.plot(smoothed_acc, 'b', label='train smooth')
            plt.legend(loc='lower right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('acc')
            plt.savefig('train_acc.png')
            plt.close()

            smoothed_ce = [np.mean(ce_batch[max(0, i - 10):i + 1]) for i in range(len(ce_batch))]
            plt.figure(figsize=(12, 8))
            plt.plot(ce_batch, 'b.', label='train')
            plt.plot(smoothed_ce, 'b', label='train smooth')
            plt.legend(loc='upper right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('ce')
            plt.savefig('train_ce.png')
            plt.close()

            smoothed_r = [np.mean(r_batch[max(0, i - 10):i + 1]) for i in range(len(r_batch))]
            plt.figure(figsize=(12, 8))
            plt.plot(r_batch, 'b.', label='train')
            plt.plot(smoothed_r, 'b', label='train smooth')
            plt.legend(loc='lower right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('r')
            plt.savefig('train_r.png')
            plt.close()

            smoothed_racc = [np.mean(racc_batch[max(0, i - 10):i + 1]) for i in range(len(racc_batch))]
            plt.figure(figsize=(12, 8))
            plt.plot(racc_batch, 'b.', label='train')
            plt.plot(smoothed_racc, 'b', label='train smooth')
            plt.legend(loc='lower right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('r acc')
            plt.savefig('train_racc.png')
            plt.close()
