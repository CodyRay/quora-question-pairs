import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

import csv
import codecs
import numpy as np
import pandas as pd
import pickle as p

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import os

if len(sys.argv) != 2:
    raise Exception("Provide .h5 file as argument")

STAMP = os.path.splitext(os.path.basename(sys.argv[1]))[0]
BASE_DIR = '../input/'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
MAX_SEQUENCE_LENGTH = 30

print(STAMP)
bst_model_weights_path = STAMP + '.h5'
bst_model_path = STAMP + '.model.h5'

model = load_model(bst_model_path)
model.summary()

model.load_weights(bst_model_weights_path)

with open(STAMP + '.tokenizer.p', 'rb') as tokenizer_p:
    tokenizer = p.load(tokenizer_p)

all_words = set(tokenizer.word_index.keys())


def remove_words(text):  # Removes stop words and words not in word2vec
    return " ".join([w for w in text.split() if w in all_words])


print("Loading Training Data")
texts_1 = []
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for i, values in enumerate(reader):
        texts_1.append(remove_words(values[3]))
        texts_2.append(remove_words(values[4]))
        labels.append(int(values[5]))

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)

print("Predicting Training Data")
trainPreds = model.predict([data_1, data_2], batch_size=2000, verbose=1)
trainPreds += model.predict([data_2, data_1], batch_size=2000, verbose=1)
trainPreds /= 2

# Recalculate this because I didn't save it!
accuracy = sum((1 for i, p in enumerate(trainPreds) if np.round(p) == labels[i])) / len(trainPreds)

print("")
print("Accuracy: {}".format(accuracy))

print("Loading Testing Data")
test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(remove_words(values[1]))
        test_texts_2.append(remove_words(values[2]))
        test_ids.append(values[0])

test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

print("Predicting Testing Data")
preds = model.predict([test_data_1, test_data_2], batch_size=2000, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=2000, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
submission.to_csv(STAMP + '.submission_{}.csv'.format(accuracy), index=False)
