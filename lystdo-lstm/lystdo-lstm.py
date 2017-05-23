import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import codecs
import pickle
import numpy as np
from nltk.corpus import stopwords

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

seed = np.random.randint(0, 10000)
print("Seed: {}".format(seed))
np.random.seed(seed)

########################################
# set directories and parameters
########################################
BASE_DIR = './'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
# whether to re-weight classes to fit the 17.5% share in test set
re_weight = True

STAMP = 'lstm_%d_%d_%.2f_%.2f_seed_%d' % (num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, seed)

print(STAMP)

########################################
# index word vectors
########################################
print('Indexing word vectors')

# How to convert glove files into this format...
# python -m gensim.scripts.glove2word2vec --input=glove.6B.300d.txt --output=glove.6B.300d.w2v.txt
# ```
# from gensim.models.keyedvectors import KeyedVectors

# model = KeyedVectors.load_word2vec_format('glove.6B.300d.w2v.txt', binary=False)
# model.save_word2vec_format('glove.6B.300d.w2v.bin', binary=True)
# ```
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,
                                             binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
# process texts in datasets
########################################
print('Processing text dataset')


def remove_words(text):  # Removes stop words and words not in word2vec
    stops = set(stopwords.words("english"))
    return " ".join([w for w in text.split() if w not in stops and w in word2vec.vocab])


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
print('Found %s texts in train.csv' % len(texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2)

with open(STAMP + '.tokenizer.p', 'wb') as tokenizer_p:
    pickle.dump(tokenizer, tokenizer_p)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

########################################
# prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in list(word_index.items()):
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' %
      np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
# sample train/validation data
########################################
# np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1) * (1 - VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val == 0] = 1.309028344

########################################
# define the model structure
########################################
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm,
                  recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
# add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################
# train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input],
              outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
# model.summary()
print(STAMP)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_weights_path = STAMP + '.h5'
bst_model_path = STAMP + '.model.h5'
model_weights_checkpoint = ModelCheckpoint(
    bst_model_weights_path, save_best_only=True, save_weights_only=True)
model_checkpoint = ModelCheckpoint(
    bst_model_path, save_best_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train,
                 validation_data=([data_1_val, data_2_val], labels_val, weight_val),
                 epochs=400, batch_size=1024, shuffle=True,
                 class_weight=class_weight,
                 callbacks=[early_stopping, model_checkpoint, model_weights_checkpoint])

model.load_weights(bst_model_weights_path)
bst_val_score = min(hist.history['val_loss'])
print("Best Value Loss = {}".format(bst_val_score))
