import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

dt = 0.16

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
    

root = '/ecg/'

mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
plt.figure(figsize=(8, 8))
for mu_string in mus_strings:
    ppo_val_T = np.load(root+'ppo/mu='+mu_string+'/T_val.npy')
    ppo_val_acc = np.load(root+'ppo/mu='+mu_string+'/acc_val.npy')
    
    ppo_val_T, ppo_val_acc = remove_worse(ppo_val_T, ppo_val_acc)
    ppo_val_T, ppo_val_acc = order(ppo_val_T, ppo_val_acc)
    plt.plot(ppo_val_T*dt, ppo_val_acc, 'o-', label=mu_string)
plt.title('ppo (val)')
plt.xlabel('T')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.show()

mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
plt.figure(figsize=(8, 8))
for mu_string in mus_strings:
    larm_val_T = np.load(root+'larm/mu='+mu_string+'/val_T.npy')
    larm_val_acc = np.load(root+'larm/mu='+mu_string+'/val_acc.npy')
    
    larm_val_T, larm_val_acc = remove_worse(larm_val_T, larm_val_acc)
    larm_val_T, larm_val_acc = order(larm_val_T, larm_val_acc)
    plt.plot(larm_val_T*dt, larm_val_acc, 'o-', label=mu_string)
plt.title('larm (val)')
plt.xlabel('T')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.show()

mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
plt.figure(figsize=(8, 8))
for mu_string in mus_strings:
    sil_val_T = np.load(root+'cis/mu='+mu_string+'/validation_n.npy')
    sil_val_acc = np.load(root+'cis/mu='+mu_string+'/validation_acc.npy')
    
    sil_val_T, sil_val_acc = remove_worse(sil_val_T, sil_val_acc)
    sil_val_T, sil_val_acc = order(sil_val_T, sil_val_acc)
    plt.plot(sil_val_T*dt, sil_val_acc, 'o-', label=mu_string)
plt.title('sil (val)')
plt.xlabel('T')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.show()



ppo_val_T = []
ppo_val_acc = []
mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
for mu_string in mus_strings:
    ppo_val_T.extend(np.load(root+'ppo/mu='+mu_string+'/T_val.npy') )
    ppo_val_acc.extend(np.load(root+'ppo/mu='+mu_string+'/acc_val.npy') )
ppo_val_T, ppo_val_acc = remove_worse(ppo_val_T, ppo_val_acc)
ppo_val_T, ppo_val_acc = order(ppo_val_T, ppo_val_acc)
    
larm_val_T = []
larm_val_acc = []
mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
for mu_string in mus_strings:
    larm_val_T.extend(np.load(root+'larm/mu='+mu_string+'/val_T.npy'))
    larm_val_acc.extend(np.load(root+'larm/mu='+mu_string+'/val_acc.npy') )
larm_val_T, larm_val_acc = remove_worse(larm_val_T, larm_val_acc)
larm_val_T, larm_val_acc = order(larm_val_T, larm_val_acc)

sil_val_T = []
sil_val_acc = []
mus_strings = ['1e-3', '3e-3', '5e-3', '7e-3', '1e-2', '3e-2', '5e-2', '7e-2', '1e-1']
for mu_string in mus_strings:
    sil_val_T.extend(np.load(root+'cis/mu='+mu_string+'/validation_n.npy'))
    sil_val_acc.extend(np.load(root+'cis/mu='+mu_string+'/validation_acc.npy') )
sil_val_T, sil_val_acc = remove_worse(sil_val_T, sil_val_acc)
sil_val_T, sil_val_acc = order(sil_val_T, sil_val_acc)

ppo_val_T *= dt
larm_val_T *= dt
sil_val_T *= dt

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



TCirc = np.load(root+'cis/mu=3e-2/validation_n.npy')[274]
accCirc = np.load(root+'cis/mu=3e-2/validation_acc.npy')[274]
plt.plot(TCirc*dt, accCirc, 'ro', markersize=30, fillstyle='none', markeredgewidth=3)  

TCirc = np.load(root+'larm/mu=7e-3/val_T.npy')[466]
accCirc = np.load(root+'larm/mu=7e-3/val_acc.npy')[466]
plt.plot(TCirc*dt, accCirc, 'ro', markersize=30, fillstyle='none', markeredgewidth=3)  

# TCirc = np.load(root+'ppo/mu=1e-3/T_val.npy')[1]
# accCirc = np.load(root+'ppo/mu=1e-3/acc_val.npy')[1]
# plt.plot(TCirc*dt, accCirc, 'ro', markersize=30, fillstyle='none', markeredgewidth=3)  








  
# plt.title('aucs: sil/larm=%.3f, '%(silAUC/larmAUC) + 'sil/ppo=%.3f'%(silAUC/ppoAUC), fontsize=20)
plt.xlabel('Mean T (seconds)', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc='center right', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('paretoAUC.png')
plt.savefig('paretoAUC.eps')
# plt.close()
plt.show()




