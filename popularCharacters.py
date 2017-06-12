import os
import csv
import codecs
from collections import Counter

dir = os.path.dirname(__file__)
BASE_DIR = os.path.join(dir, './input/')
TRAIN_DATA_FILE = os.path.join(BASE_DIR, 'train.csv')
TEST_DATA_FILE = os.path.join(BASE_DIR, 'test.csv')


def iter():
    with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for values in reader:
            yield values[3]
            yield values[4]

    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for values in reader:
            yield values[1]
            yield values[2]


def main():
    with codecs.open('popular.csv', 'w', encoding='utf-8') as f:
        characters = Counter((c for line in iter() for c in line))

        total = sum(characters.values())
        writer = csv.writer(f, )
        writer.writerow(['Char', 'Count', 'Share'])
        for n, c in reversed(sorted(((n, c) for c, n in characters.items()))):
            writer.writerow([c, n, n / total])


if __name__ == "__main__":
    main()
