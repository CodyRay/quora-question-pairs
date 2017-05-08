import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import codecs
import pickle
import numpy as np
from nltk.corpus import stopwords
import pandas as pd

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

dir = os.path.dirname(__file__)
BASE_DIR = os.path.join(dir, '../input/')
EMBEDDING_FILE = os.path.join(BASE_DIR, 'GoogleNews-vectors-negative300.bin')
TRAIN_DATA_FILE = os.path.join(BASE_DIR, 'train.csv')
TEST_DATA_FILE = os.path.join(BASE_DIR, 'test.csv')
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
# whether to re-weight classes to fit the 17.5% share in test set
RE_WEIGHT = True


def seed(s=None):
    s = np.random.randint(0, 10000) if s is None else s
    print("Seed: {}".format(s))
    np.random.seed(s)
    return s


def correction(word, allwords):
    '''https://www.kaggle.com/cpmpml/spell-checker-using-word2vec'''
    def P(word):
        if word not in allwords:
            return -float('inf')
        "Probability of `word`."
        # use inverse of rank as proxy
        return - allwords[word]

    def candidates(word):
        "Generate possible spelling corrections for word."
        return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

    def known(words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in allwords)

    def edits1(word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in edits1(word) for e2 in edits1(e1))

    "Most probable spelling correction for word."
    return max(candidates(word), key=P)


def clean_words(text, allwords, corrections, discarding):  # Removes stop words and words not in allwords
    stops = set(stopwords.words("english"))
    letters = set('abcdefghijklmnopqrstuvwxyz ')

    def remove_words(words):
        for w in words:
            if w not in stops and w in allwords:
                yield w
            elif w not in stops:
                (c, cw) = corrections.get(w, (1, None))
                if cw is None:
                    cw = correction(w, allwords)
                if cw in allwords:
                    corrections[w] = (c + 1, cw)
                    yield cw
                else:
                    discarding[w] = discarding.get(w, 0) + 1

    text = "".join([c for c in text.lower() if c in letters])
    return " ".join(remove_words(text.split()))


def load_training_text(stamp, allwords):
    corrections = dict()
    discarding = dict()

    texts_1 = []
    texts_2 = []
    labels = []
    with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for i, values in enumerate(reader):
            texts_1.append(clean_words(values[3], allwords, corrections, discarding))
            texts_2.append(clean_words(values[4], allwords, corrections, discarding))
            labels.append(int(values[5]))

    print('Found %s texts in train.csv' % len(texts_1))

    with open(os.path.join(dir, stamp + '.training_badwords.txt'), 'w') as badWords:
        for w, (c, cw) in corrections:
            print("Corrected {}: {} to {}".format(c, w, cw), file=badWords)
        for w, c in discarding:
            print("Discarding {}: {}".format(c, w), file=badWords)

    return (texts_1, texts_2, np.array(labels))


def load_testing_text(stamp, allwords):
    corrections = dict()
    discarding = dict()

    print("Loading Testing Data")
    test_texts_1 = []
    test_texts_2 = []
    test_ids = []
    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for i, values in enumerate(reader):
            if i % 99 == 0:
                print(i + 1)
            test_texts_1.append(clean_words(values[1], allwords, corrections, discarding))
            test_texts_2.append(clean_words(values[2], allwords, corrections, discarding))
            test_ids.append(values[0])

    print('Found %s texts in test.csv' % len(test_texts_1))

    with open(os.path.join(dir, stamp + '.test_badwords.txt'), 'w') as badWords:
        for w, (c, cw) in corrections:
            print("Corrected {}: {} to {}".format(c, w, cw), file=badWords)
        for w, c in discarding:
            print("Discarding {}: {}".format(c, w), file=badWords)

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
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))

    allwords = dict()
    for i, word in enumerate(word2vec.index2word):
        allwords[word] = i

    return word2vec, allwords


def make_tokenizer(stamp, texts):
    print("Building Tokenizer")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)

    print('Found %s unique tokens' % len(tokenizer.word_index))

    tpath = os.path.join(dir, stamp + '.tokenizer.p')
    print("Writting Tokenizer to " + tpath)
    with open(tpath, 'wb') as tokenizer_p:
        pickle.dump(tokenizer, tokenizer_p)


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
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
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

    return data_1_train, data_2_train, labels_train, data_1_val, data_2_val, labels_val


def main():
    s = seed()

    num_lstm = np.random.randint(175, 275)
    num_dense = np.random.randint(100, 150)
    rate_drop_lstm = 0.15 + np.random.rand() * 0.25
    rate_drop_dense = 0.15 + np.random.rand() * 0.25

    act = 'relu'

    stamp = 'lstm_%d_%d_%.2f_%.2f_seed_%d' % (num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, s)

    print(stamp)

    word2vec, allwords = load_word2vec()

    print('Processing text dataset')

    (texts_1, texts_2, labels) = load_training_text(stamp, allwords)

    tokenizer = make_tokenizer(stamp, texts_1 + texts_2)

    data_1 = texts_to_data(tokenizer, texts_1)
    data_2 = texts_to_data(tokenizer, texts_2)
    print('Shape of data tensor:', data_1.shape)
    print('Shape of label tensor:', labels.shape)

    nb_words, embedding_matrix = make_embedding_matrix(tokenizer, word2vec)

    d1_train, d2_train, labels_train, d1_val, d2_val, labels_val, weight_val = split_training(data_1, data_2, labels)

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
    if RE_WEIGHT:
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
    print(stamp)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_weights_path = os.path.join(dir, stamp + '.h5')
    bst_model_path = os.path.join(dir, stamp + '.model.h5')
    model_weights_checkpoint = ModelCheckpoint(
        bst_model_weights_path, save_best_only=True, save_weights_only=True)
    model_checkpoint = ModelCheckpoint(
        bst_model_path, save_best_only=True)

    hist = model.fit([d1_train, d2_train], labels_train,
                     validation_data=([d1_val, d2_val], labels_val, weight_val),
                     epochs=400, batch_size=1024, shuffle=True,
                     class_weight=class_weight,
                     callbacks=[early_stopping, model_checkpoint, model_weights_checkpoint])

    model.load_weights(bst_model_weights_path)
    bst_val_score = min(hist.history['val_loss'])
    print("Best Value Loss = {}".format(bst_val_score))

    # Submission
    test_texts_1, test_texts_2, test_ids = load_testing_text(stamp, allwords)

    test_data_1 = texts_to_data(test_texts_1)
    test_data_2 = texts_to_data(test_texts_2)

    print("Predicting Testing Data")
    preds = model.predict([test_data_1, test_data_2], batch_size=2000, verbose=1)
    preds += model.predict([test_data_2, test_data_1], batch_size=2000, verbose=1)
    preds /= 2

    submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
    submission.to_csv(os.path.join(dir, stamp + '.submission_{}.csv'.format(bst_val_score)), index=False)


if __name__ == "__main__":
    main()