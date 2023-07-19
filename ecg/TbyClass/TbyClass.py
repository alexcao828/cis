import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dt = 0.16

SILvalAllTs = np.load('/ecg/cis/mu=3e-2/valAllTs274.npy') * dt
LARMvalAllTs = np.load('/ecg/larm/mu=7e-3/valAllTs466.npy') * dt
PPOvalAllTs = np.load('/ecg/ppo/mu=1e-3/valAllTs1.npy') * dt

valY = np.load('y_val.npy', allow_pickle=True)[0:len(SILvalAllTs)]

whereNORM = np.where(valY == 'NORM')[0]
whereCD = np.where(valY == 'CD')[0]
whereHYP = np.where(valY == 'HYP')[0]
whereMI = np.where(valY == 'MI')[0]
whereSTTC = np.where(valY == 'STTC')[0]

# print('NORM')
# print(np.mean(valAllTs[whereNORM]))

# print('CD')
# print(np.mean(valAllTs[whereCD]))

# print('HYP')
# print(np.mean(valAllTs[whereHYP]))

# print('MI')
# print(np.mean(valAllTs[whereMI]))

# print('STTC')
# print(np.mean(valAllTs[whereSTTC]))



a1 = pd.DataFrame({ 'Diagnosis' : np.repeat('NORM',len(SILvalAllTs[whereNORM])), 'T': SILvalAllTs[whereNORM], 'model': np.repeat('CIS',len(SILvalAllTs[whereNORM]))})
a2 = pd.DataFrame({ 'Diagnosis' : np.repeat('NORM',len(LARMvalAllTs[whereNORM])), 'T': LARMvalAllTs[whereNORM], 'model': np.repeat('LARM',len(LARMvalAllTs[whereNORM]))})
# a3 = pd.DataFrame({ 'Diagnosis' : np.repeat('NORM',len(PPOvalAllTs[whereNORM])), 'T': PPOvalAllTs[whereNORM], 'model': np.repeat('PPO',len(PPOvalAllTs[whereNORM]))})

b1 = pd.DataFrame({ 'Diagnosis' : np.repeat('CD',len(SILvalAllTs[whereCD])), 'T': SILvalAllTs[whereCD], 'model': np.repeat('CIS',len(SILvalAllTs[whereCD]))})
b2 = pd.DataFrame({ 'Diagnosis' : np.repeat('CD',len(LARMvalAllTs[whereCD])), 'T': LARMvalAllTs[whereCD], 'model': np.repeat('LARM',len(LARMvalAllTs[whereCD]))})
# b3 = pd.DataFrame({ 'Diagnosis' : np.repeat('CD',len(PPOvalAllTs[whereCD])), 'T': PPOvalAllTs[whereCD], 'model': np.repeat('PPO',len(PPOvalAllTs[whereCD]))})

c1 = pd.DataFrame({ 'Diagnosis' : np.repeat('HYP',len(SILvalAllTs[whereHYP])), 'T': SILvalAllTs[whereHYP], 'model': np.repeat('CIS',len(SILvalAllTs[whereHYP]))})
c2 = pd.DataFrame({ 'Diagnosis' : np.repeat('HYP',len(LARMvalAllTs[whereHYP])), 'T': LARMvalAllTs[whereHYP], 'model': np.repeat('LARM',len(LARMvalAllTs[whereHYP]))})
# c3 = pd.DataFrame({ 'Diagnosis' : np.repeat('HYP',len(PPOvalAllTs[whereHYP])), 'T': PPOvalAllTs[whereHYP], 'model': np.repeat('PPO',len(PPOvalAllTs[whereHYP]))})

d1 = pd.DataFrame({ 'Diagnosis' : np.repeat('MI',len(SILvalAllTs[whereMI])), 'T': SILvalAllTs[whereMI], 'model': np.repeat('CIS',len(SILvalAllTs[whereMI]))})
d2 = pd.DataFrame({ 'Diagnosis' : np.repeat('MI',len(LARMvalAllTs[whereMI])), 'T': LARMvalAllTs[whereMI], 'model': np.repeat('LARM',len(LARMvalAllTs[whereMI]))})
# d3 = pd.DataFrame({ 'Diagnosis' : np.repeat('MI',len(PPOvalAllTs[whereMI])), 'T': PPOvalAllTs[whereMI], 'model': np.repeat('PPO',len(PPOvalAllTs[whereMI]))})

e1 = pd.DataFrame({ 'Diagnosis' : np.repeat('STTC',len(SILvalAllTs[whereSTTC])), 'T': SILvalAllTs[whereSTTC], 'model': np.repeat('CIS',len(SILvalAllTs[whereSTTC]))})
e2 = pd.DataFrame({ 'Diagnosis' : np.repeat('STTC',len(LARMvalAllTs[whereSTTC])), 'T': LARMvalAllTs[whereSTTC], 'model': np.repeat('LARM',len(LARMvalAllTs[whereSTTC]))})
# e3 = pd.DataFrame({ 'Diagnosis' : np.repeat('STTC',len(PPOvalAllTs[whereSTTC])), 'T': PPOvalAllTs[whereSTTC], 'model': np.repeat('PPO',len(PPOvalAllTs[whereSTTC]))})

df=a1.append(a2).append(b1).append(b2).append(c1).append(c2).append(d1).append(d2).append(e1).append(e2)
# df=a1.append(a2).append(a3).append(b1).append(b2).append(b3).append(c1).append(c2).append(c3).append(d1).append(d2).append(d3).append(e1).append(e2).append(e3)
 
# Usual boxplot
plt.figure(figsize=(8, 8))
sns.boxplot(x='Diagnosis', y='T', hue='model', data=df, palette={"CIS": "tab:orange", "LARM": "tab:green"})
plt.xlabel( "Diagnosis" , size = 20 )
plt.ylabel( "T" , size = 20 )
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.gca().legend().set_title('')
plt.legend(fontsize=15)
plt.savefig('TbyClass.png')
plt.savefig('TbyClass.eps')
plt.show()


# np.quantile(SILvalAllTs[whereSTTC], 0.75)
