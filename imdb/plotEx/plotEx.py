import numpy as np
import matplotlib.pyplot as plt

SILvalActProbs = np.load('/imdb/cis/mu=1e-3/allValActProbs27.npy')
SILvalClassProbs = np.load('/imdb/cis/mu=1e-3/allValClassProbs27.npy')

LARMvalActProbs = np.load('/imdb/larm/mu=1e-3/allValActProbs189.npy')
LARMvalClassProbs = np.load('/imdb/larm/mu=1e-3/allValClassProbs189.npy')
                            
PPOvalActProbs = np.load('/imdb/ppo/mu=1e-3/allValActProbs6.npy')
PPOvalClassProbs = np.load('/imdb/ppo/mu=1e-3/allValClassProbs6.npy')


xVal = np.load('x_validation.npy')
yVal = np.load('y_validation.npy')

dictionary = np.load('imdb_dictionary.npy', allow_pickle='TRUE').item()

samples = [621, 837]
# samples = np.arange(5235, 5255)
ending = [20, 60]
# ending = 60*np.ones(20).astype(int)
for i in range(len(samples)):
    sample = samples[i]
    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(1,ending[i]+1), PPOvalActProbs[sample, 0:ending[i]], '-',color='tab:blue', linewidth=3, label='PPO') 
    plt.plot(np.arange(1,ending[i]+1), LARMvalActProbs[sample, 0:ending[i]], '-',color='tab:green', linewidth=3, label='LARM') 
    plt.plot(np.arange(1,ending[i]+1), SILvalActProbs[sample, 0:ending[i]], '-',color='tab:orange', linewidth=3, label='CIS') 
    # plt.plot(np.arange(1,ending[i]+1), valActProbs[sample, 0:ending[i]]*0+0.5, 'r--', linewidth=3) 
    plt.ylabel('Prob. stop', fontsize=15)
    if sample == 621:
        plt.xticks(np.arange(2,22,2), np.arange(2,22,2), fontsize=15)
    else:
        plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='center left', fontsize=15)
    
    plt.subplot(3, 1, 2)
    # T = np.where(valActProbs[sample] >= 0.5)[0]
    # T = T[0]
    if yVal[sample] == 1:
        plt.plot(np.arange(1,ending[i]+1), SILvalClassProbs[sample, 0:ending[i]], '-',color='tab:orange', linewidth=3)  
        plt.plot(np.arange(1,ending[i]+1), LARMvalClassProbs[sample, 0:ending[i]], '-',color='tab:green', linewidth=3)  
        plt.plot(np.arange(1,ending[i]+1), PPOvalClassProbs[sample, 0:ending[i]], '-',color='tab:blue', linewidth=3)  
        # plt.plot(T+1, valClassProbs[sample, T], 'ro', markersize=15, fillstyle='none')
        plt.ylabel('Prob. + review', fontsize=15)
    else:
        plt.plot(np.arange(1,ending[i]+1), 1-SILvalClassProbs[sample, 0:ending[i]], '-',color='tab:orange', linewidth=3)  
        plt.plot(np.arange(1,ending[i]+1), 1-LARMvalClassProbs[sample, 0:ending[i]], '-',color='tab:green', linewidth=3)  
        plt.plot(np.arange(1,ending[i]+1), 1-PPOvalClassProbs[sample, 0:ending[i]], '-',color='tab:blue', linewidth=3)  
        # plt.plot(T+1, 1-valClassProbs[sample, T], 'ro', markersize=15, fillstyle='none')
        plt.ylabel('Prob. - review', fontsize=15)
    plt.ylim(0.08, 1.02)

    sentence = '[1] '
    for j in range(ending[i]):
        word = dictionary[xVal[sample, j]]
        if word == '<pad>':
            break
        sentence += word
        if (j+1) % 10 == 0 and j < ending[i]-1:
            sentence += '\n' + '[' + str(j+2) + '] '
        else:
            sentence += ' '
    plt.xlabel(sentence, fontsize=15)
    if sample == 621:
        plt.xticks(np.arange(2,22,2), np.arange(2,22,2), fontsize=15)
    else:
        plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('valEx' + str(sample) + '.png')
    plt.savefig('valEx' + str(sample) + '.eps')
    plt.show()
