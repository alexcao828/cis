import re
# import nltk
import pandas as pd
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def process(x):
    x = re.sub('[,\.!?:()"]', '', x)
    x = re.sub('<.*?>', ' ', x)
    x = re.sub('http\S+', ' ', x)
    x = re.sub('[^a-zA-Z0-9]', ' ', x)
    x = re.sub('\s+', ' ', x)
    return x.lower().strip()

# def sw_remove(x):
#     sw_set = set(nltk.corpus.stopwords.words('english'))
#     words = nltk.tokenize.word_tokenize(x)
#     filtered_list = [word for word in words if word not in sw_set]
#     return ' '.join(filtered_list)

def load_dataset():
    df = pd.read_csv('IMDB Dataset.csv')
    x_data = df['review']       # Reviews/Input
    y_data = df['sentiment']    # Sentiment/Output

    # PRE-PROCESS REVIEW
    x_data = x_data.apply(lambda x: process(x))
    # x_data = x_data.apply(lambda x: sw_remove(x))
    
    # ENCODE SENTIMENT -> 0 & 1
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    return x_data, y_data

def get_mean_length(x):
    review_length = []
    for review in x:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))

x_data, y_data = load_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)
x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size = 0.5)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_validation = np.asarray(y_validation)

# ENCODE REVIEW
token = Tokenizer(lower=False)    # no need lower, because already lowered the data in load_data()
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)
x_validation = token.texts_to_sequences(x_validation)

mean_length = get_mean_length(x_train)

x_train = pad_sequences(x_train, maxlen=mean_length, padding='post', truncating='post')
x_validation = pad_sequences(x_validation, maxlen=mean_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=mean_length, padding='post', truncating='post')

np.save('x_train.npy', x_train)
np.save('x_validation.npy', x_validation)
np.save('x_test.npy', x_test)

np.save('y_train.npy', y_train)
np.save('y_validation.npy', y_validation)
np.save('y_test.npy', y_test)

old_imdb_dictionary = token.word_index
new_imdb_dictionary = dict([(value, key) for key, value in old_imdb_dictionary.items()])
new_imdb_dictionary[0] = '<pad>'
np.save('imdb_dictionary.npy', new_imdb_dictionary)
