import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab

dir = os.path.dirname(__file__)
BASE_DIR = os.path.join(dir, './input/')


def main():
    def proccess(source, destination):
        source = os.path.join(BASE_DIR, source)
        destination = os.path.join(BASE_DIR, destination)
        print("\n\nProccessing {} to {}".format(source, destination))
        word2vec = load_word2vec(source)
        cleanWord2vec = cleanCopy(word2vec, destination + '.cleaning.log')
        cleanWord2vec.save_word2vec_format(destination, binary=True)
        load_word2vec(destination)  # To see how many load
        print("Finished Proccessing {} to {}".format(source, destination))

    proccess('glove.6B.50d.w2v.bin', 'glove.6B.50d.w2v.clean.bin')
    proccess('glove.6B.100d.w2v.bin', 'glove.6B.100d.w2v.clean.bin')
    proccess('glove.6B.200d.w2v.bin', 'glove.6B.200d.w2v.clean.bin')
    proccess('glove.6B.300d.w2v.bin', 'glove.6B.300d.w2v.clean.bin')
    proccess('glove.42B.300d.w2v.bin', 'glove.42B.300d.w2v.clean.bin')
    proccess('glove.840B.300d.w2v.bin', 'glove.840B.300d.w2v.clean.bin')
    proccess('GoogleNews-vectors-negative300.bin', 'GoogleNews-vectors-negative300.clean.bin')


def load_word2vec(file):
    print('Loading {}'.format(file))
    # How to convert glove files into this format...
    # python -m gensim.scripts.glove2word2vec --input=glove.6B.300d.txt --output=glove.6B.300d.w2v.txt
    # ```
    # from gensim.models.keyedvectors import KeyedVectors

    # model = KeyedVectors.load_word2vec_format('glove.6B.300d.w2v.txt', binary=False)
    # model.save_word2vec_format('glove.6B.300d.w2v.bin', binary=True)
    # ```
    word2vec = KeyedVectors.load_word2vec_format(file, binary=True)
    print('Loaded %s word vectors of word2vec' % len(word2vec.vocab))

    return word2vec


def cleanCopy(word2vec, logfile):
    letters = set('abcdefghijklmnopqrstuvwxyz')

    def clean(w):
        return "".join([c for c in w.lower() if c in letters])

    vector_size = word2vec.syn0.shape[1]
    vocab_size = len(word2vec.vocab)
    result = KeyedVectors()
    result.vector_size = vector_size
    result.syn0 = np.zeros((vocab_size, vector_size), dtype=np.float32)

    def add_word(word, weights):
        word_id = len(result.vocab)
        result.vocab[word] = Vocab(index=word_id, count=vocab_size - word_id)
        result.syn0[word_id] = weights
        result.index2word.append(word)

    with open(logfile, 'w') as log:
        print("{}, {}".format('Original', 'Saved'), file=log)
        for word in word2vec.index2word:
            cleanWord = clean(word)
            # Only add a word if it isn't already there.
            # If the same clean word is in the original twice (i.e. Trump and trump)
            # then the most popular one will be used since index2word contains the words in
            # decreasing frequency
            if cleanWord != '' and cleanWord not in result.vocab:
                if word != cleanWord:
                    try:
                        print("{}, {}".format(word, cleanWord), file=log)
                    except UnicodeEncodeError:
                        pass  # UnicodeEncodeError: 'charmap' codec can't encode character '\u0142' in position 0: character maps to <undefined>
                weights = word2vec.word_vec(word)
                add_word(cleanWord, weights)
            else:
                try:
                    print("{}, {}".format(word, '<>'), file=log)
                except UnicodeEncodeError:
                    pass  # UnicodeEncodeError: 'charmap' codec can't encode character '\u0142' in position 0: character maps to <undefined>

    if result.syn0.shape[0] != len(result.vocab):
        result.syn0 = np.ascontiguousarray(result.syn0[: len(result.vocab)])

    return result


if __name__ == "__main__":
    main()
