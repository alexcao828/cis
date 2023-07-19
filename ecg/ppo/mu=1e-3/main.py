import numpy as np
import tensorflow as tf
from policy_net import policy_net
from ppo import ppo_train
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random

batch_size = 128
num_actions = 2
num_classes = 5

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
num_training_episodes = len(y_train)

T_end = np.shape(x_train)[1]
num_frequencies = np.shape(x_train)[2]
num_leads = np.shape(x_train)[3]

num_times_thru_training_set = 100
num_iterations = int(num_training_episodes * num_times_thru_training_set)
M = int(num_training_episodes/batch_size)

x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')
num_val_episodes =  int(len(y_val))

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
num_test_episodes =  int(len(y_test))

dict = {
  "NORM": 0,
  "MI": 1,
  "STTC": 2,
  "HYP": 3,
  "CD": 4
}
y_train = np.asarray([dict[diag] for diag in y_train])
y_val = np.asarray([dict[diag] for diag in y_val])
y_test = np.asarray([dict[diag] for diag in y_test])

policy = policy_net('policy', T_end, num_frequencies, num_leads)
old_policy = policy_net('old_policy', T_end, num_frequencies, num_leads)
ppo = ppo_train(policy, old_policy)

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

    T_val = []
    acc_val = []
    ce_val = []
    r_val = []
    racc_val = []
    rtime_val = []
    valAllTs = np.zeros((num_times_thru_training_set, num_val_episodes))

    T_test = []
    acc_test = []
    ce_test = []
    r_test = []
    racc_test = []
    rtime_test = []

    indices = np.arange(num_training_episodes)
    np.random.shuffle(indices)
    counter = 0
    epoch = 0

    for ep in range(1, num_iterations + 1):
        i = indices[counter]

        spectrograms = x_train[i]
        action_probs = policy.get_action_prob(spectrograms)

        c = 0
        complete = False
        # Run through each episode
        while complete == False:
            temp = action_probs[c]
            action = np.random.choice(action_space, p=temp)
            c += 1
            if action == 1 or c == T_end:
                complete = True

        y = np.zeros(num_classes)
        y[int(y_train[i])] = 1

        pe_grads, guess, ce, r, racc, rtime = ppo.get_grads(spectrograms,y, c * np.ones(1), np.asarray([c - 1, 1]), 'train')

        for j, g in enumerate(pe_grads):
            grad_buffer_pe[j] =grad_buffer_pe[j]+ g

        T_ep.append(c)
        y_ep.append(y_train[i])
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
            T_temp = []
            y_temp = []
            guess_temp = []
            ce_temp = []
            r_temp = []
            racc_temp = []
            rtime_temp = []
            valTs = []

            for v in range(num_val_episodes):
                spectrograms = x_val[v]
                action_probs = policy.get_action_prob(spectrograms)

                c = 0
                complete = False
                # Run through each episode
                while complete == False:
                    temp = action_probs[c]
                    action = np.random.choice(action_space, p=temp)
                    c += 1
                    if action == 1 or c == T_end:
                        complete = True

                y = np.zeros(num_classes)
                y[int(y_val[v])] = 1

                guess, ce, r, racc, rtime = ppo.get_grads(spectrograms, y, c * np.ones(1), np.asarray([c - 1, 1]), 'val/test')

                T_temp.append(c)
                y_temp.append(y_val[v])
                guess_temp.append(guess)
                ce_temp.append(ce)
                r_temp.append(r)
                racc_temp.append(racc)
                rtime_temp.append(rtime)
                valTs.append(c)

            T_val.append(np.mean(np.asarray(T_temp)))
            acc_val.append(accuracy_score(y_temp, guess_temp))
            ce_val.append(np.mean(np.asarray(ce_temp)))
            r_val.append(np.mean(np.asarray(r_temp)))
            racc_val.append(np.mean(np.asarray(racc_temp)))
            rtime_val.append(np.mean(np.asarray(rtime_temp)))
            valAllTs[epoch, :] = valTs

            np.save('T_val.npy', T_val)
            np.save('acc_val.npy', acc_val)
            np.save('ce_val.npy', ce_val)
            np.save('r_val.npy', r_val)
            np.save('racc_val.npy', racc_val)
            np.save('rtime_val.npy', rtime_val)
            np.save('valAllTs' + str(epoch) + '.npy', valAllTs[epoch])

            epoch += 1

            "test"
            T_temp = []
            y_temp = []
            guess_temp = []
            ce_temp = []
            r_temp = []
            racc_temp = []
            rtime_temp = []

            for t in range(num_test_episodes):
                spectrograms = x_test[t]
                action_probs = policy.get_action_prob(spectrograms)

                c = 0
                complete = False
                # Run through each episode
                while complete == False:
                    temp = action_probs[c]
                    action = np.random.choice(action_space, p=temp)
                    c += 1
                    if action == 1 or c == T_end:
                        complete = True

                y = np.zeros(num_classes)
                y[int(y_test[t])] = 1

                guess, ce, r, racc, rtime = ppo.get_grads(spectrograms, y, c * np.ones(1), np.asarray([c - 1, 1]), 'val/test')

                T_temp.append(c)
                y_temp.append(y_test[t])
                guess_temp.append(guess)
                ce_temp.append(ce)
                r_temp.append(r)
                racc_temp.append(racc)
                rtime_temp.append(rtime)

            T_test.append(np.mean(np.asarray(T_temp)))
            acc_test.append(accuracy_score(y_temp, guess_temp))
            ce_test.append(np.mean(np.asarray(ce_temp)))
            r_test.append(np.mean(np.asarray(r_temp)))
            racc_test.append(np.mean(np.asarray(racc_temp)))
            rtime_test.append(np.mean(np.asarray(rtime_temp)))

            np.save('T_test.npy', T_test)
            np.save('acc_test.npy', acc_test)
            np.save('ce_test.npy', ce_test)
            np.save('r_test.npy', r_test)
            np.save('racc_test.npy', racc_test)
            np.save('rtime_test.npy', rtime_test)

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
            if len(T_val) > 0:
                plt.plot(M*np.arange(1,len(T_val)+1), T_val, 'gs', label='val')
                plt.plot(M * np.arange(1, len(T_test) + 1), T_test, 'rs', label='test')
            plt.legend(loc='upper right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('T')
            plt.savefig('T.png')
            plt.close()

            smoothed_acc = [np.mean(acc_batch[max(0, i - 10):i + 1]) for i in range(len(acc_batch))]
            plt.figure(figsize=(12, 8))
            plt.plot(acc_batch, 'b.', label='train')
            plt.plot(smoothed_acc, 'b', label='train smooth')
            if len(acc_val) > 0:
                plt.plot(M*np.arange(1,len(acc_val)+1), acc_val, 'gs', label='val')
                plt.plot(M * np.arange(1, len(acc_test) + 1), acc_test, 'rs', label='test')
            plt.legend(loc='lower right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('acc')
            plt.savefig('acc.png')
            plt.close()

            smoothed_ce = [np.mean(ce_batch[max(0, i - 10):i + 1]) for i in range(len(ce_batch))]
            plt.figure(figsize=(12, 8))
            plt.plot(ce_batch, 'b.', label='train')
            plt.plot(smoothed_ce, 'b', label='train smooth')
            if len(ce_val) > 0:
                plt.plot(M*np.arange(1,len(ce_val)+1), ce_val, 'gs', label='val')
                plt.plot(M * np.arange(1, len(ce_test) + 1), ce_test, 'rs', label='test')
            plt.legend(loc='upper right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('ce')
            plt.savefig('ce.png')
            plt.close()

            smoothed_r = [np.mean(r_batch[max(0, i - 10):i + 1]) for i in range(len(r_batch))]
            plt.figure(figsize=(12, 8))
            plt.plot(r_batch, 'b.', label='train')
            plt.plot(smoothed_r, 'b', label='train smooth')
            if len(r_val) > 0:
                plt.plot(M*np.arange(1,len(r_val)+1), r_val, 'gs', label='val')
                plt.plot(M * np.arange(1, len(r_test) + 1), r_test, 'rs', label='test')
            plt.legend(loc='lower right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('r')
            plt.savefig('r.png')
            plt.close()

            smoothed_racc = [np.mean(racc_batch[max(0, i - 10):i + 1]) for i in range(len(racc_batch))]
            plt.figure(figsize=(12, 8))
            plt.plot(racc_batch, 'b.', label='train')
            plt.plot(smoothed_racc, 'b', label='train smooth')
            if len(racc_val) > 0:
                plt.plot(M*np.arange(1,len(racc_val)+1), racc_val, 'gs', label='val')
                plt.plot(M * np.arange(1, len(racc_test) + 1), racc_test, 'rs', label='test')
            plt.legend(loc='lower right', prop={'size': 20})
            plt.xlabel('mb')
            plt.ylabel('r acc')
            plt.savefig('racc.png')
            plt.close()
