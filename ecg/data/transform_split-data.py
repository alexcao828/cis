import pandas as pd
import numpy as np
import wfdb
import ast
import random
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split 

path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index and y_dic[key] == 100.0:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))
# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

def randChoice(classes):
    if len(classes) == 0:
        return 'REMOVE'
    else:
        return random.choice(classes)
Y['diagnostic_superclass_randChoice'] = Y.diagnostic_superclass.apply(randChoice)

plt.figure()
plt.hist(Y['diagnostic_superclass_randChoice'])
plt.savefig('diagnostic_superclass_hist.png')
plt.close()

y = Y.diagnostic_superclass_randChoice.values
keep = []
for i in range(len(y)):
    if y[i] != 'REMOVE':
        keep.append(i)
keep = np.asarray(keep)
x = np.take(X, keep, axis=0)
y = np.take(y, keep)

f, t, Sxx = signal.spectrogram(x, fs=sampling_rate, nperseg=32, noverlap=16, axis=1)
Sxx = np.transpose(Sxx,[0,3,1,2])
Sxx = abs(Sxx)
mask = Sxx > 0
Sxx[mask] = np.log(Sxx[mask])

plt.figure()
plt.imshow(Sxx[0, :, :, 0])
plt.savefig('example_spectrogram.png')
plt.close()

x_train, x_test, y_train, y_test = train_test_split(Sxx, y, test_size = 0.2)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5)
np.save('x_train.npy', x_train)
np.save('x_val.npy', x_val)
np.save('x_test.npy', x_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

np.save('t.npy', t)

