import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import codecs
import pickle
import numpy as np
import pandas as pd
import shutil

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GRU, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

dir = os.path.dirname(__file__)
BASE_DIR = os.path.join(dir, '../input/')
EMBEDDING_FILE = os.path.join(BASE_DIR, 'glove.840B.300d.w2v.clean.bin')
EMBEDDING_DIM = 300
TRAIN_DATA_FILE = os.path.join(BASE_DIR, 'train.csv')
TEST_DATA_FILE = os.path.join(BASE_DIR, 'test.csv')
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
VALIDATION_SPLIT = 0.1
# whether to re-weight classes to fit the 17.5% share in test set
RE_WEIGHT = True
STOPPING_PATIENCE = 3
ACT = 'relu'


def seed(s=None):
    s = np.random.randint(0, 10000) if s is None else s
    print("Seed: {}".format(s))
    np.random.seed(s)
    return s


def clean_words(text, allwords, discarding):
    letters = set('abcdefghijklmnopqrstuvwxyz')

    def remove_letters(s):
        for c in s:
            if c in letters:
                yield c
            elif "'" == c:
                pass
            else:
                yield ' '

    def remove_words(words):
        for w in words:
            if w in allwords:
                yield w
            if w == 'youve':
                yield 'you'
                yield 'have'
            if w == 'shouldnt':
                yield 'should'
                yield 'not'
            else:
                discarding[w] = discarding.get(w, 0) + 1

    text = "".join(remove_letters(text.lower()))
    return " ".join(remove_words(text.split()))


def load_training_text(stamp, allwords):
    discarding = dict()
    print("Loading Training Data")

    texts_1 = []
    texts_2 = []
    labels = []
    with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for i, values in enumerate(reader):
            fancyProgressBar(i, 404290)
            texts_1.append(clean_words(values[3], allwords, discarding))
            texts_2.append(clean_words(values[4], allwords, discarding))
            labels.append(int(values[5]))
        fancyProgressBarClear(True)

    print('Loaded %s texts in train.csv' % len(texts_1))

    p = os.path.join(dir, stamp, 'train_badwords.csv')
    print("Writting bad words to " + p)
    with open(p, 'w') as badWords:
        for w, c in discarding.items():
            print("{:3}, {}".format(c, w), file=badWords)

    return (texts_1, texts_2, np.array(labels))


def load_testing_text(stamp, allwords):
    discarding = dict()

    print("Loading Testing Data")
    test_texts_1 = []
    test_texts_2 = []
    test_ids = []
    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for i, values in enumerate(reader):
            fancyProgressBar(i, 2345796)
            test_texts_1.append(clean_words(values[1], allwords, discarding))
            test_texts_2.append(clean_words(values[2], allwords, discarding))
            test_ids.append(values[0])
        fancyProgressBarClear(True)

    print('Loaded %s texts in test.csv' % len(test_texts_1))

    p = os.path.join(dir, stamp, 'test_badwords.csv')
    print("Writting bad words to " + p)
    with open(p, 'w') as badWords:
        for w, c in discarding.items():
            print("{:3}, {}".format(c, w), file=badWords)

    return (test_texts_1, test_texts_2, np.array(test_ids))


def load_word2vec():
    print('Loading {}'.format(EMBEDDING_FILE))
    # How to convert glove files into this format...
    # python -m gensim.scripts.glove2word2vec --input=glove.6B.300d.txt --output=glove.6B.300d.w2v.txt
    # ```
    # from gensim.models.keyedvectors import KeyedVectors

    # model = KeyedVectors.load_word2vec_format('glove.6B.300d.w2v.txt', binary=False)
    # model.save_word2vec_format('glove.6B.300d.w2v.bin', binary=True)
    # ```
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    print('Loaded %s word vectors of word2vec' % len(word2vec.vocab))

    allwords = dict()
    for i, word in enumerate(word2vec.index2word):
        allwords[word] = i

    return word2vec, allwords


def make_tokenizer(stamp, texts):
    print("Building Tokenizer")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)

    print('Found %s unique tokens' % len(tokenizer.word_index))

    tpath = os.path.join(dir, stamp, 'tokenizer.p')
    print("Writting Tokenizer to " + tpath)
    with open(tpath, 'wb') as tokenizer_p:
        pickle.dump(tokenizer, tokenizer_p)

    return tokenizer


def texts_to_data(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


def make_embedding_matrix(tokenizer, word2vec):
    print('Preparing embedding matrix')

    nb_words = min(MAX_NB_WORDS, len(tokenizer.word_index)) + 1

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in list(tokenizer.word_index.items()):
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)

    return nb_words, embedding_matrix


def split_training(data_1, data_2, labels):
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
    if RE_WEIGHT:
        weight_val *= 0.472001959
        weight_val[labels_val == 0] = 1.309028344

    return data_1_train, data_2_train, labels_train, data_1_val, data_2_val, labels_val, weight_val


def build_model(nb_words, embedding_matrix, num_gru, num_dense, rate_drop_gru, rate_drop_dense):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    gru_layer = GRU(num_gru, dropout=rate_drop_gru,
                    recurrent_dropout=rate_drop_gru)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = gru_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = gru_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=ACT)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    return model


def model_fit(model, stamp, d1_train, d2_train, labels_train, d1_val, d2_val, labels_val, weight_val):
    bst_model_weights_path = os.path.join(dir, stamp, 'weights.h5')
    bst_model_path = os.path.join(dir, stamp, 'model.h5')

    early_stopping = EarlyStopping(monitor='val_loss', patience=STOPPING_PATIENCE)
    model_weights_checkpoint = ModelCheckpoint(bst_model_weights_path, save_best_only=True, save_weights_only=True)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True)

    if RE_WEIGHT:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None

    hist = model.fit([d1_train, d2_train], labels_train,
                     validation_data=([d1_val, d2_val], labels_val, weight_val),
                     epochs=20, batch_size=128, shuffle=True,
                     class_weight=class_weight,
                     callbacks=[early_stopping, model_checkpoint, model_weights_checkpoint])

    model.load_weights(bst_model_weights_path)
    return hist


def model_prediction(model, test_data_1, test_data_2):
    preds = model.predict([test_data_1, test_data_2], batch_size=128, verbose=1)
    preds += model.predict([test_data_2, test_data_1], batch_size=128, verbose=1)
    preds /= 2
    return preds


def main():
    s = seed()

    # Original Params
    # num_gru = np.random.randint(175, 275)
    # num_dense = np.random.randint(100, 150)
    # rate_drop_gru = 0.15 + np.random.rand() * 0.25
    # rate_drop_dense = 0.15 + np.random.rand() * 0.25

    num_gru = np.random.randint(225, 275)
    num_dense = np.random.randint(100, 125)
    rate_drop_gru = 0.25 + np.random.rand() * 0.25
    rate_drop_dense = 0.25 + np.random.rand() * 0.25

    stamp = 'gru_%d_%d_%.2f_%.2f_seed_%d' % (num_gru, num_dense, rate_drop_gru, rate_drop_dense, s)

    print(stamp)
    os.mkdir(stamp)
    shutil.copyfile(__file__, os.path.join(dir, stamp, os.path.basename(__file__)))

    word2vec, allwords = load_word2vec()

    (texts_1, texts_2, labels) = load_training_text(stamp, allwords)

    tokenizer = make_tokenizer(stamp, texts_1 + texts_2)

    data_1 = texts_to_data(tokenizer, texts_1)
    data_2 = texts_to_data(tokenizer, texts_2)

    nb_words, embedding_matrix = make_embedding_matrix(tokenizer, word2vec)

    d1_train, d2_train, labels_train, d1_val, d2_val, labels_val, weight_val = split_training(data_1, data_2, labels)

    model = build_model(nb_words, embedding_matrix, num_gru, num_dense, rate_drop_gru, rate_drop_dense)

    hist = model_fit(model, stamp, d1_train, d2_train, labels_train, d1_val, d2_val, labels_val, weight_val)
    bst_val_score = min(hist.history['val_loss'])
    print("Best Value Loss = {}".format(bst_val_score))

    history_p = os.path.join(dir, stamp, 'history_{}.p'.format(bst_val_score))
    print("Writing History to {}".format(history_p))
    with open(history_p, 'wb') as history_f:
        pickle.dump(tokenizer, history_f)

    # Submission
    test_texts_1, test_texts_2, test_ids = load_testing_text(stamp, allwords)

    test_data_1 = texts_to_data(tokenizer, test_texts_1)
    test_data_2 = texts_to_data(tokenizer, test_texts_2)

    print("Predicting Testing Data")
    preds = model_prediction(model, test_data_1, test_data_2)

    submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
    submission.to_csv(os.path.join(dir, stamp, 'submission_{}.csv'.format(bst_val_score)), index=False)


lastProgress = [] if sys.stdout.isatty() else None


def fancyProgressBar(complete, total=100, title="", width=79):
    if lastProgress is None:
        return
    barSize = width - 8 - len(title)
    barValue = int(barSize * complete / total)
    value = int(100 * complete / total)
    if lastProgress == [title, width, barValue, value]:
        return
    lastProgress[:] = [title, width, barValue, value]
    bar = "".join((("#" if barValue >= x else "-") for x in range(barSize)))
    print("\r{} [{:3}%] {}".format(title, value, bar), end="")


def fancyProgressBarClear(clear=False):
    if lastProgress is None:
        return
    width = lastProgress[1]
    lastProgress[:] = [None]
    if clear:
        print("\r".ljust(width + 1), end='')
        print("\r", end='')
    else:
        print()


if __name__ == "__main__":
    main()
