"""
Using the vsm model to pass the embedding from the high similarity items entity
to the unseen item entity.
"""
import json
import argparse
import pickle as pickle
from math import pi, acos
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.analyse
from annoy import AnnoyIndex

PARSER = argparse.ArgumentParser()
PARSER.add_argument("unseen_event_file",
                    type=str,
                    help="The unseen events title list")
PARSER.add_argument("embedding_file",
                    type=str,
                    help="The embedding json file")
PARSER.add_argument("corpus_file",
                    type=str,
                    help="The items' title text file (json) for training vsm")
ARGS = PARSER.parse_args()
UNSEEN_EVENTS_FILE = ARGS.unseen_event_file
EMBEDDING_FILE = ARGS.embedding_file
CORPUS_FILE = ARGS.corpus_file
jieba.set_dictionary("./jieba-zh_TW_NEPT_src/jieba/dict.txt")
def vsm(fp=CORPUS_FILE):
    with open(fp, 'r') as json_file_in:
        item_tags_dict = json.load(json_file_in)
        index_id_dict = {}
        corpus = []
        for index, (id_key, tags) in enumerate(item_tags_dict.items()):
            sentence = [tag for tag, weight in tags]
            corpus.append(" ".join(sentence))
            index_id_dict[index] = id_key
        vectorizer = TfidfVectorizer()
        document_term_matrix = vectorizer.fit_transform(corpus)
        dim = document_term_matrix.shape[1]
        annoy_index = AnnoyIndex(dim)
        for index, vector in enumerate(document_term_matrix):
            annoy_index.add_item(index, vector.toarray()[0])
        annoy_index.build(10) # 10 trees
        annoy_index.save('vsm_tfidf.ann')
        return index_id_dict, vectorizer, document_term_matrix

def closest_topK(unseen_event, ids_dict, model, dim, topK=10):
    unseen_even_tags = jieba.analyse.extract_tags(unseen_event)
    unseen_event_vector = [0] * dim
    for tag in unseen_even_tags:
        if tag in model.vocabulary_:
            unseen_event_vector[model.vocabulary_[tag]] += model.idf_[model.vocabulary_[tag]]
    annoy_index = AnnoyIndex(dim)
    annoy_index.load('vsm_tfidf.ann')
    ranking_list = annoy_index.get_nns_by_vector(unseen_event_vector, 10, search_k=-1, include_distances=True)
    propgation_list = []
    for matrix_row, score in zip(ranking_list[0], ranking_list[1]):
        propgation_list.append((ids_dict[matrix_row], score))
    return propgation_list

def embedding_propgation(ranking_list, weight_func = lambda x : 1, fp=EMBEDDING_FILE):
    with open(EMBEDDING_FILE, 'r') as json_file_in:
        embedding_dict = json.load(json_file_in)
    accumulate_vector = []
    accumulate_weight = 0
    weight_list = []
    add_count = 0
    for ranking_list_index, (id_, score) in enumerate(ranking_list):
        try:
            added_vector = embedding_dict[id_]
        except KeyError:
            # Due to some events are lack of people book them,
            # they are removed from the training set.
            print("{} is not a significant event so that not included in the training embedding.".format(id_))
            continue
        weight = weight_func(score)
        weight_list.append(weight)
        if add_count == 0:
            accumulate_vector = list(map(lambda x: x * weight, added_vector))
        else:
            for index, (element1, element2) in\
                            enumerate(zip(accumulate_vector, added_vector)):
                accumulate_vector[index] = element1 + element2 * weight
        add_count += 1
        accumulate_weight += weight
    print('weight list: {}'.format(list(map(lambda x: x / accumulate_weight, weight_list))))
    print('{} related events.'.format(add_count))
    return list(map(lambda x: x / accumulate_weight, accumulate_vector))

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


if __name__ == "__main__":
    IDS_DICT, TRAINED_MODEL, DOC_MATRIX = vsm()
    #pickle.dump(TRAINED_MODEL, open('vsm_model.pickle', 'wb'))
    UNSEEN_DICT = load_unseen()
    UNSEEN_EMBEDDING_DICT = {}
    for id_, (title_string, description) in UNSEEN_DICT.items():
        print('unssenId:', id_)
        query_string = title_string + description
        ID_LIST =\
            closest_topK(query_string, IDS_DICT, TRAINED_MODEL, DOC_MATRIX.shape[1])
        print(ID_LIST)
        UNSEEN_EMBEDDING_DICT[id_] = embedding_propgation(ID_LIST, weight_func=lambda x : 0.0001 + x)
        print()
    with open('unssen_events_rep_hpe(tfidf_2018unseen_top100queries_strong_user_before2018).txt', 'wt') as fout:
        fout.write("{}\n".format(len(UNSEEN_EMBEDDING_DICT)))
        for id_, embedding in UNSEEN_EMBEDDING_DICT.items():
            fout.write("{} {}\n".format(id_, ' '.join(map(lambda x:str(round(x, 6)),embedding))))
