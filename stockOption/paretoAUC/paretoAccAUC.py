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
    ppo_val_acc = np.load(root+'ppo/mu='+mu_string+'/val_acc_mean.npy')
    
    ppo_val_T, ppo_val_acc = remove_worse(ppo_val_T, ppo_val_acc)
    ppo_val_T, ppo_val_acc = order(ppo_val_T, ppo_val_acc)
    plt.plot(ppo_val_T, ppo_val_acc, 'o-', label=mu_string)
plt.title('ppo (val)')
plt.xlabel('T')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.show()

mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
plt.figure(figsize=(8, 8))
for mu_string in mus_strings:
    larm_val_T = np.load(root+'larm/mu='+mu_string+'/val_T_mean.npy')
    larm_val_acc = np.load(root+'larm/mu='+mu_string+'/val_acc_mean.npy')
    
    larm_val_T, larm_val_acc = remove_worse(larm_val_T, larm_val_acc)
    larm_val_T, larm_val_acc = order(larm_val_T, larm_val_acc)
    plt.plot(larm_val_T, larm_val_acc, 'o-', label=mu_string)
plt.title('larm (val)')
plt.xlabel('T')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.show()

mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
plt.figure(figsize=(8, 8))
for mu_string in mus_strings:
    sil_val_T = np.load(root+'cis/mu='+mu_string+'/val_T.npy')
    sil_val_acc = np.load(root+'cis/mu='+mu_string+'/val_acc.npy')
    
    sil_val_T, sil_val_acc = remove_worse(sil_val_T, sil_val_acc)
    sil_val_T, sil_val_acc = order(sil_val_T, sil_val_acc)
    plt.plot(sil_val_T, sil_val_acc, 'o-', label=mu_string)
plt.title('sil (val)')
plt.xlabel('T')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.show()



ppo_val_T = []
ppo_val_acc = []
mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
for mu_string in mus_strings:
    ppo_val_T.extend(np.load(root+'ppo/mu='+mu_string+'/val_T_mean.npy') )
    ppo_val_acc.extend(np.load(root+'ppo/mu='+mu_string+'/val_acc_mean.npy') )
ppo_val_T, ppo_val_acc = remove_worse(ppo_val_T, ppo_val_acc)
ppo_val_T, ppo_val_acc = order(ppo_val_T, ppo_val_acc)
    
larm_val_T = []
larm_val_acc = []
mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
for mu_string in mus_strings:
    larm_val_T.extend(np.load(root+'larm/mu='+mu_string+'/val_T_mean.npy'))
    larm_val_acc.extend(np.load(root+'larm/mu='+mu_string+'/val_acc_mean.npy') )
larm_val_T, larm_val_acc = remove_worse(larm_val_T, larm_val_acc)
larm_val_T, larm_val_acc = order(larm_val_T, larm_val_acc)

sil_val_T = []
sil_val_acc = []
mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
for mu_string in mus_strings:
    sil_val_T.extend(np.load(root+'cis/mu='+mu_string+'/val_T.npy'))
    sil_val_acc.extend(np.load(root+'cis/mu='+mu_string+'/val_acc.npy') )
sil_val_T, sil_val_acc = remove_worse(sil_val_T, sil_val_acc)
sil_val_T, sil_val_acc = order(sil_val_T, sil_val_acc)

right = np.amax([ppo_val_T[-1], larm_val_T[-1], sil_val_T[-1]])
ppo_val_T, ppo_val_acc = piecewiseConstant(ppo_val_T, ppo_val_acc, right)
larm_val_T, larm_val_acc = piecewiseConstant(larm_val_T, larm_val_acc , right)
sil_val_T, sil_val_acc = piecewiseConstant(sil_val_T, sil_val_acc, right)

ppoAUC = metrics.auc(ppo_val_T, ppo_val_acc)
larmAUC = metrics.auc(larm_val_T, larm_val_acc)
silAUC = metrics.auc(sil_val_T, sil_val_acc)



plt.figure(figsize=(8, 8))
plt.plot(ppo_val_T, ppo_val_acc, '-', label='PPO (AUC=%.2f)'%ppoAUC, color='tab:blue', linewidth=3)  
plt.plot(larm_val_T, larm_val_acc, '-', label='LARM (AUC=%.2f)'%larmAUC, color='tab:green', linewidth=3) 
plt.plot(sil_val_T, sil_val_acc, '-', label='CIS (AUC=%.2f)'%silAUC, color='tab:orange', linewidth=3)


TCirc = np.load(root+'cis/mu=1e-2/val_T.npy')[1223]
accCirc = np.load(root+'cis/mu=1e-2/val_acc.npy')[1223]
plt.plot(TCirc, accCirc, 'ro', markersize=30, fillstyle='none', markeredgewidth=3)    

TCirc = np.load(root+'larm/mu=3e-3/val_T_mean.npy')[69]
accCirc = np.load(root+'larm/mu=3e-3/val_acc_mean.npy')[69]
plt.plot(TCirc, accCirc, 'ro', markersize=30, fillstyle='none', markeredgewidth=3)   

TCirc = np.load(root+'ppo/mu=5e-2/val_T_mean.npy')[3]
accCirc = np.load(root+'ppo/mu=5e-2/val_acc_mean.npy')[3]
plt.plot(TCirc, accCirc, 'ro', markersize=30, fillstyle='none', markeredgewidth=3)    





   
# plt.title('aucs: sil/larm=%.3f, '%(silAUC/larmAUC) + 'sil/ppo=%.3f'%(silAUC/ppoAUC), fontsize=20)
plt.xlabel('Mean T (days)', fontsize=20)
plt.ylabel('Mean accuracy', fontsize=20)
plt.legend(loc='lower right', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('paretoAccAUC.png')
plt.savefig('paretoAccAUC.eps')
# plt.close()
plt.show()




