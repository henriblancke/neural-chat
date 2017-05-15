from __future__ import print_function

import spacy
import gensim
from tqdm import tqdm
from variables import params
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

GO_ID = 3000000
PAD_ID = 3000001
EOS_ID = 3000002
UNK_ID = 3000003

special_map = [_GO, _PAD, _EOS, _UNK, _ORG, _URL, _EMAIL, _PERSON, _EVENT, _PRODUCT, _LOCATION, _FACULTY]

print("Loading spaCy and Word2Vec models...")
nlp = spacy.load('en')
model = gensim.models.Word2Vec.load_word2vec_format(params.embeddings_dir, binary=True)


def embeddings_matrix():
    return model.syn0


def tokenizer(sentence):
    doc = nlp(sentence)

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


def specials_to_id(token):
    base = 3000000
    if token in special_map:
        uid = special_map.index(token)
        index = base + uid
    else:
        uid = special_map.index(_UNK)
        index = base + uid

    return index


def id_to_specials(uid):
    index = int(str(uid)[-2:])
    try:
        return special_map[index]
    except IndexError:
        return _UNK


def sentence_to_ids(sentence):
    sent_ids = []

    sentence = unicode(sentence, errors='ignore')
    for token in tokenizer(sentence):
        try:
            vocab = model.vocab[token]
            sent_ids.append(vocab.index)
        except KeyError:
            index = specials_to_id(token)
            sent_ids.append(index)

    return sent_ids


def ids_to_sentence(ids):
    words = []
    for uid in ids:
        try:
            word = model.index2word[uid]
            words.append(word)
        except IndexError:
            word = id_to_specials(uid)
            words.append(word)

    return words


def data_to_token_ids(data_path, target_path):
    if not gfile.Exists(target_path):
        with gfile.GFile(data_path, mode='r') as data_file:
            with gfile.GFile(target_path, mode='w') as tokens_file:
                for sentence in tqdm(data_file):
                    token_ids = sentence_to_ids(sentence)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, *args):
    # Points to training and dev preprocess
    train_path = data_dir + "/training_data"
    dev_path = data_dir + "/validation_data"

    print("Preparing training preprocess...")
    train_ids_path = train_path + ".ids"
    data_to_token_ids(train_path, train_ids_path)

    print("Preparing validation preprocess...")
    dev_ids_path = dev_path + ".ids"
    data_to_token_ids(dev_path, dev_ids_path)

    return train_ids_path, dev_ids_path, None
