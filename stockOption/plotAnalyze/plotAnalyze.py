import numpy as np
import matplotlib.pyplot as plt

num_time_steps = 30

val_close = np.load('val_close.npy')
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
closings = valCloseReshaped


SILactProbs= np.load('/stockOption/cis/mu=1e-2/actProbs1223.npy')
LARMactProbs= np.load('/stockOption/larm/mu=3e-3/actProbs69.npy')
PPOactProbs= np.load('/stockOption/ppo/mu=5e-2/actProbs3.npy')




        
plt.figure(figsize=(8, 8))
for  i in [2, 10, 22, 36, 42]:
    plt.subplot(2,1,1)
    plt.plot(np.arange(1,31), 100*(closings[i, :]- closings[i, 0])/ closings[i, 0] , 'k-', linewidth=2)
    plt.plot(np.arange(1,31), 10*np.ones(30), 'k--')
    plt.plot(np.arange(1,31), -10*np.ones(30), 'k--')
    plt.ylim([-20, 20])
    plt.ylabel('Closing price % change', size=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.subplot(2,1,2)
    if i == 42:
        plt.plot(np.arange(1,31), PPOactProbs[i], '-', color='tab:blue', linewidth=2, label='PPO')
        plt.plot(np.arange(1,31), LARMactProbs[i], '-', color='tab:green', linewidth=2, label='LARM')
        plt.plot(np.arange(1,31), SILactProbs[i], '-', color='tab:orange', linewidth=2, label='CIS')
        plt.plot(np.arange(1,31), SILactProbs[i]*0+0.5, 'k--')
    else:
        plt.plot(np.arange(1,31), SILactProbs[i], '-', color='tab:orange', linewidth=2)
        plt.plot(np.arange(1,31), SILactProbs[i]*0+0.5, 'k--')
        plt.plot(np.arange(1,31), LARMactProbs[i], '-', color='tab:green', linewidth=2)
        plt.plot(np.arange(1,31), PPOactProbs[i], '-', color='tab:blue', linewidth=2)
    plt.xlabel('Time (days)', size=15)
    plt.ylabel('Prob. stop', size=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
plt.tight_layout()
plt.savefig('fastExamples.png')
plt.savefig('fastExamples.eps')
plt.show()

plt.figure(figsize=(8, 8))
for  i in [4, 5, 6, 8, 12]:
    plt.subplot(2,1,1)
    plt.plot(np.arange(1,31), 100*(closings[i, :]- closings[i, 0])/ closings[i, 0] , 'k-', linewidth=2)
    plt.plot(np.arange(1,31), 10*np.ones(30), 'k--')
    plt.plot(np.arange(1,31), -10*np.ones(30), 'k--')
    plt.ylim([-20, 20])
    plt.ylabel('Closing price % change', size=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.subplot(2,1,2)
    if i == 12:
        plt.plot(np.arange(1,31), PPOactProbs[i], '-', color='tab:blue', linewidth=2, label='PPO')
        plt.plot(np.arange(1,31), LARMactProbs[i], '-', color='tab:green', linewidth=2, label='LARM')
        plt.plot(np.arange(1,31), SILactProbs[i], '-', color='tab:orange', linewidth=2, label='CIS')
    else:
        plt.plot(np.arange(1,31), SILactProbs[i], '-', color='tab:orange', linewidth=2)
        plt.plot(np.arange(1,31), LARMactProbs[i], '-', color='tab:green', linewidth=2)
        plt.plot(np.arange(1,31), PPOactProbs[i], '-', color='tab:blue', linewidth=2)
    plt.plot(np.arange(1,31), SILactProbs[i]*0+0.5, 'k--')
    plt.xlabel('Time (days)', size=15)
    plt.ylabel('Prob. stop', size=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
plt.tight_layout()
plt.savefig('slowExamples.png')
plt.savefig('slowExamples.eps')
plt.show()



