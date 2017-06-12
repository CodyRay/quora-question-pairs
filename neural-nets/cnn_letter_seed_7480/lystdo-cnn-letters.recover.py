import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import codecs
import pickle
import numpy as np
import pandas as pd

from keras.models import load_model

dir = os.path.dirname(__file__)
BASE_DIR = os.path.join(dir, '../../input/')
TEST_DATA_FILE = os.path.join(BASE_DIR, 'test.csv')
MAX_LENGTH = 150  # Quora's official character limit, some are longer though
CHARACTERS = set(' 0123456789\'-"&(),./:?[]+abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
CHARACTER_PADDING = '_'
CHARACTER_OTHER = '#'
CHARACTERS_TO_TOKENS = {c: i for i, c in enumerate([CHARACTER_PADDING, CHARACTER_OTHER] + list(CHARACTERS))}
BATCH_SIZE = 256


def clean_text(text):
    builder = []
    for i in range(MAX_LENGTH):
        if len(text) > i:
            if text[i] in CHARACTERS:
                builder.append(text[i])
            else:
                builder.append(CHARACTER_OTHER)
        else:
            builder.append(CHARACTER_PADDING)
    return builder


def load_testing_text():
    print("Loading Testing Data")
    test_texts_1 = []
    test_texts_2 = []
    test_ids = []
    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for i, values in enumerate(reader):
            fancyProgressBar(i, 2345796)
            test_texts_1.append(clean_text(values[1]))
            test_texts_2.append(clean_text(values[2]))
            test_ids.append(values[0])
        fancyProgressBarClear(True)

    print('Loaded %s texts in test.csv' % len(test_texts_1))
    return (np.array(test_texts_1), np.array(test_texts_2), np.array(test_ids))


def texts_to_data(texts):
    return np.array([[CHARACTERS_TO_TOKENS[c] for c in text] for text in texts])


def model_prediction(model, test_data_1, test_data_2):
    preds = model.predict([test_data_1, test_data_2], batch_size=BATCH_SIZE, verbose=1)
    preds += model.predict([test_data_2, test_data_1], batch_size=BATCH_SIZE, verbose=1)
    preds /= 2
    return preds


def main():
    bst_model_weights_path = os.path.join(dir, 'weights.h5')
    bst_model_path = os.path.join(dir, 'model.h5')
    model = load_model(bst_model_path)
    model.load_weights(bst_model_weights_path)

    model.summary()

    history_p = os.path.join(dir, 'history_0.3190652870542765.p')
    print("Reading History from {}".format(history_p))
    with open(history_p, 'rb') as history_f:
        hist_history = pickle.load(history_f)

    bst_val_score = min(hist_history['val_loss'])

    # Submission
    test_texts_1, test_texts_2, test_ids = load_testing_text()

    test_data_1 = texts_to_data(test_texts_1)
    test_data_2 = texts_to_data(test_texts_2)

    print("Predicting Testing Data")
    preds = model_prediction(model, test_data_1, test_data_2)

    submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
    submission.to_csv(os.path.join(dir, 'submission_{}.csv'.format(bst_val_score)), index=False)


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
