"""Utilities for building vocabularies and tokenizing data.
"""
from __future__ import print_function

import os
import spacy
from tqdm import tqdm
from tensorflow.python.platform import gfile

# Special vocab symbols
_GO = "_GO"
_PAD = "_PAD"
_EOS = "_EOS"
_UNK = "_UNK"
_ORG = "_ORG"
_URL = "_URL"
_EMAIL = "_EMAIL"
_PERSON = "_PERSON"
_EVENT = "_EVENT"
_PRODUCT = "_PRODUCT"
_LOCATION = "_LOCATION"
_FACULTY = "_FACULTY"

GO_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3

_START_VOCAB = [_GO,
                _PAD,
                _EOS,
                _UNK,
                # _ORG,
                _URL,
                _EMAIL,
                # _PERSON,
                # _EVENT,
                # _PRODUCT,
                # _LOCATION,
                # _FACULTY
                ]

nlp = spacy.load('en')


def tokenizer(sentence):
    doc = nlp(unicode(sentence, errors='ignore'))

    words = []
    for token in doc:
        if token.like_url:
            words.append(_URL)
            continue
        elif token.is_digit:
            continue
        elif token.like_num:
            continue
        elif token.like_email:
            words.append(_EMAIL)
            continue
        # elif token.ent_type_ in ['GPE', 'LOC']:
        #    words.append(_LOCATION)
        #    continue
        # elif token.ent_type_ in ['FAC']:
        #    words.append(_FACULTY)
        #    continue
        # elif token.ent_type_ in ['EVENT']:
        #    words.append(_EVENT)
        #    continue
        # elif token.ent_type_ in ['PERSON']:
        #    words.append(_PERSON)
        #    continue
        # elif token.ent_type_ in ['PRODUCT']:
        #    words.append(_PRODUCT)
        #    continue
        else:
            words.append(token.text.lower())
            continue

    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
    """Create vocabulary file (if it does not exist yet) from preprocess file.

    Data file is assumed to contain one sentence per line that has been converted to all lowercase
    (except for 'I' and all associated contractions).
    Each sentence is tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: preprocess file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from preprocess %s" % (vocabulary_path, data_path))

        vocab = {}
        with gfile.GFile(data_path, mode="r") as f:

            for line in tqdm(f):

                tokens = tokenizer(line)
                for token in tokens:
                    try:
                        vocab[token] += 1
                    except:
                        vocab[token] = 1

            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)

            with gfile.GFile(vocabulary_path + 'complete', mode="w") as master_file:
                for w in sorted(vocab, key=vocab.get, reverse=True):
                    master_file.write(w + ': ' + str(vocab[w]) + '\n')

            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]

            with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):

        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: a string, the sentence to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
    words = tokenizer(sentence)

    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path):
    """Tokenize preprocess file and turn into token-ids using given vocabulary file.

    This function loads preprocess line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the preprocess file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
        if None, basic_tokenizer will be used.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing preprocess in %s" % data_path)

        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                for line in tqdm(data_file):
                    token_ids = sentence_to_token_ids(line, vocab)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, vocabulary_size):
    """Create vocabularies and tokenize preprocess.

    Args:
      data_dir: directory in which the preprocess sets will be stored.
      vocabulary_size: size of the vocabulary to create and use.

    Returns:
      A tuple of 3 elements:
        (1) path to the token-ids for training preprocess-set,
        (2) path to the token-ids for development preprocess-set,
        (3) path to the vocabulary file
    """
    # Points to training and dev preprocess
    train_path = data_dir + "/training_data"
    dev_path = data_dir + "/validation_data"

    # Create vocabulary of appropriate size
    vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
    create_vocabulary(vocab_path, train_path, vocabulary_size)

    # Create token ids for the training preprocess.
    train_ids_path = train_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(train_path, train_ids_path, vocab_path)

    # Create token ids for the development preprocess.
    dev_ids_path = dev_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(dev_path, dev_ids_path, vocab_path)

    return train_ids_path, dev_ids_path, vocab_path
