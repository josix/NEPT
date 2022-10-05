import json
import argparse

import numpy as np
import implicit
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import pinv

PARSER = argparse.ArgumentParser()
PARSER.add_argument("unseen_event_file",
                    type=str,
                    help="The unseen events title list")
PARSER.add_argument("corpus_file",
                    type=str,
                    help="The items' title text file (json) for training vsm")
ARGS = PARSER.parse_args()
UNSEEN_EVENTS_FILE = ARGS.unseen_event_file
CORPUS_FILE = ARGS.corpus_file
jieba.set_dictionary("./jieba-zh_TW_NEPT_src/jieba/dict.txt")

def load_unseen(fp=UNSEEN_EVENTS_FILE):
    with open(fp, 'rt') as fin:
        unseen_dict = {}
        for line in fin:
            splitted_line = line.strip().split(',')
            if len(splitted_line) == 1:
                continue
            id_, title, description = splitted_line
            unseen_dict[id_] = (title, description)
        return unseen_dict

def train_mf(fp=CORPUS_FILE):
    with open(fp, 'r') as json_file_in:
        item_to_terms = json.load(json_file_in)
        corpus = []
        for id_key, terms in item_to_terms.items():
            sentence = [term for term, weight in terms]
            corpus.append(" ".join(sentence))
        vectorizer = TfidfVectorizer()
        document_term_matrix = vectorizer.fit_transform(corpus)
    print('document_term_matrix: ', document_term_matrix.shape)
    model = implicit.als.AlternatingLeastSquares(factors=128, iterations=30, calculate_training_loss=True, num_threads=20)
    model.fit(document_term_matrix)
    print('item_factors_matrix: ', model.item_factors.shape)
    print('term_factors_matrix: ', model.user_factors.shape)
    term_factor_matrix = model.user_factors
    return vectorizer, term_factor_matrix

def get_unseen_tfidf(unseen_event, model, dim):
    unseen_event_terms = jieba.analyse.extract_tags(unseen_event)
    unseen_event_vector = [0] * dim
    for term in unseen_event_terms:
        if term in model.vocabulary_:
            unseen_event_vector[model.vocabulary_[term]] += model.idf_[model.vocabulary_[term]]
    return unseen_event_vector

if __name__ == "__main__":
    VSM_MODEL, TERM_FACTOR_MATRIX = train_mf()
    UNSEENID_TO_SENTENCE = load_unseen()
    UNSEEN_EMBEDDING_DICT = {}
    for id_, (title_string, description) in UNSEENID_TO_SENTENCE.items():
        newitem_sentence = title_string + description
        unseen_tfidf = np.array([get_unseen_tfidf(newitem_sentence, VSM_MODEL, TERM_FACTOR_MATRIX.shape[0])])
        UNSEEN_EMBEDDING_DICT[id_] = np.dot(unseen_tfidf, pinv(TERM_FACTOR_MATRIX.T)).tolist()[0]

    with open('mf.txt', 'wt') as fout:
        fout.write(f"{len(UNSEEN_EMBEDDING_DICT)}\n")
        for id_, embedding in UNSEEN_EMBEDDING_DICT.items():
            fout.write(f"{id_} {' '.join(map(lambda x: str(round(x, 6)), embedding))}\n")

