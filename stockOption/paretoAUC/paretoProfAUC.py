import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def remove_worse(x, y):
    if len(x) != len(y):
        STOP
    #x-y plane, to right and lower is worse
    ind = []
    for i in range(len(x)):
        others = np.arange(len(x))
        others = np.delete(others, i)
        for j in others:
            if x[i] >= x[j] and y[i] <= y[j]:
                ind.append(i)
                break
    ind = np.unique(ind)
    x = np.delete(x, ind)
    y = np.delete(y, ind)
    return x, y

def order(x, y):
    if len(x) != len(y):
        STOP
    sort = np.argsort(x)
    x = x[sort]
    y = y[sort]
    return x, y

def piecewiseConstant(x,y, x_max):
    if len(x) != len(y):
        STOP
    x_new = []
    y_new = []
    for i in range(1, len(x)):
        x_new.append(x[i])
        y_new.append(y[i-1])
    if x[-1] < x_max:
        x_new.append(x_max)
        y_new.append(y[-1])
    
    x2 = []
    y2 = []
    x2.append(x[0])
    y2.append(y[0])
    for i in range(1, len(x)):
        x2.append(x_new[i-1])
        x2.append(x[i])
        y2.append(y_new[i-1])
        y2.append(y[i])
    if x2[-1] < x_max:
        x2.append(x_new[-1])
        y2.append(y_new[-1])

    x2 = np.asarray(x2)
    y2 = np.asarray(y2)
    return x2, y2
    

root = '/stockOption/'

mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
plt.figure(figsize=(8, 8))
for mu_string in mus_strings:
    ppo_val_T = np.load(root+'ppo/mu='+mu_string+'/val_T_mean.npy')
    ppo_val_profit = np.load(root+'ppo/mu='+mu_string+'/val_profit_mean.npy')
    
    ppo_val_T, ppo_val_profit = remove_worse(ppo_val_T, ppo_val_profit)
    ppo_val_T, ppo_val_profit = order(ppo_val_T, ppo_val_profit)
    plt.plot(ppo_val_T, ppo_val_profit, 'o-', label=mu_string)
plt.title('ppo (val)')
plt.xlabel('T')
plt.ylabel('profit')
plt.legend(loc='lower right')
plt.show()

mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
plt.figure(figsize=(8, 8))
for mu_string in mus_strings:
    larm_val_T = np.load(root+'larm/mu='+mu_string+'/val_T_mean.npy')
    larm_val_profit = np.load(root+'larm/mu='+mu_string+'/val_profit_mean.npy')
    
    larm_val_T, larm_val_profit = remove_worse(larm_val_T, larm_val_profit)
    larm_val_T, larm_val_profit = order(larm_val_T, larm_val_profit)
    plt.plot(larm_val_T, larm_val_profit, 'o-', label=mu_string)
plt.title('larm (val)')
plt.xlabel('T')
plt.ylabel('profit')
plt.legend(loc='lower right')
plt.show()

mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
plt.figure(figsize=(8, 8))
for mu_string in mus_strings:
    sil_val_T = np.load(root+'cis/mu='+mu_string+'/val_T.npy')
    sil_val_profit = np.load(root+'cis/mu='+mu_string+'/val_profit.npy')
    
    sil_val_T, sil_val_profit = remove_worse(sil_val_T, sil_val_profit)
    sil_val_T, sil_val_profit = order(sil_val_T, sil_val_profit)
    plt.plot(sil_val_T, sil_val_profit, 'o-', label=mu_string)
plt.title('sil (val)')
plt.xlabel('T')
plt.ylabel('profit')
plt.legend(loc='lower right')
plt.show()



ppo_val_T = []
ppo_val_profit = []
mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
for mu_string in mus_strings:
    ppo_val_T.extend(np.load(root+'ppo/mu='+mu_string+'/val_T_mean.npy') )
    ppo_val_profit.extend(np.load(root+'ppo/mu='+mu_string+'/val_profit_mean.npy') )
ppo_val_T, ppo_val_profit = remove_worse(ppo_val_T, ppo_val_profit)
ppo_val_T, ppo_val_profit = order(ppo_val_T, ppo_val_profit)
    
larm_val_T = []
larm_val_profit = []
mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
for mu_string in mus_strings:
    larm_val_T.extend(np.load(root+'larm/mu='+mu_string+'/val_T_mean.npy'))
    larm_val_profit.extend(np.load(root+'larm/mu='+mu_string+'/val_profit_mean.npy') )
larm_val_T, larm_val_profit = remove_worse(larm_val_T, larm_val_profit)
larm_val_T, larm_val_profit = order(larm_val_T, larm_val_profit)

sil_val_T = []
sil_val_profit = []
mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
for mu_string in mus_strings:
    sil_val_T.extend(np.load(root+'cis/mu='+mu_string+'/val_T.npy'))
    sil_val_profit.extend(np.load(root+'cis/mu='+mu_string+'/val_profit.npy') )
sil_val_T, sil_val_profit = remove_worse(sil_val_T, sil_val_profit)
sil_val_T, sil_val_profit = order(sil_val_T, sil_val_profit)

right = np.amax([ppo_val_T[-1], larm_val_T[-1], sil_val_T[-1]])
ppo_val_T, ppo_val_profit = piecewiseConstant(ppo_val_T, ppo_val_profit, right)
larm_val_T, larm_val_profit = piecewiseConstant(larm_val_T, larm_val_profit , right)
sil_val_T, sil_val_profit = piecewiseConstant(sil_val_T, sil_val_profit, right)

ppoAUC = metrics.auc(ppo_val_T, ppo_val_profit)
larmAUC = metrics.auc(larm_val_T, larm_val_profit)
silAUC = metrics.auc(sil_val_T, sil_val_profit)


plt.figure(figsize=(8, 8))
plt.plot(ppo_val_T, ppo_val_profit, '-', label='PPO (AUC=%.2f)'%ppoAUC, color='tab:blue', linewidth=3)  
plt.plot(larm_val_T, larm_val_profit, '-', label='LARM (AUC=%.2f)'%larmAUC, color='tab:green', linewidth=3) 
plt.plot(sil_val_T, sil_val_profit, '-', label='CIS (AUC=%.2f)'%silAUC, color='tab:orange', linewidth=3)    
# plt.title('aucs: sil/larm=%.3f, '%(silAUC/larmAUC) + 'sil/ppo=%.3f'%(silAUC/ppoAUC), fontsize=20)
plt.xlabel('Mean T (days)', fontsize=20)
plt.ylabel('Mean profit (dollars)', fontsize=20)
plt.legend(loc='lower right', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('paretoProfAUC.png')
plt.savefig('paretoProfAUC.eps')
# plt.close()
plt.show()




