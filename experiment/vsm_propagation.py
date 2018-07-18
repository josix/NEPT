"""
Using the vsm model to pass the embedding from the high similarity items entity
to the unseen item entity.
"""
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.analyse

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

jieba.set_dictionary("./jieba-zh_TW/jieba/dict.txt")
def vsm(fp=CORPUS_FILE):
    with open(fp, 'r') as json_file_in:
        item_tags_dict = json.load(json_file_in)
        index_id_dict = {}
        corpus = []
        for index, (id_key, value) in enumerate(item_tags_dict.items()):
            corpus.append(" ".join(value))
            index_id_dict[index] = id_key
        vectorizer = TfidfVectorizer()
        document_term_matrix = vectorizer.fit_transform(corpus)
        return index_id_dict, vectorizer, document_term_matrix

def closest_topK(unseen_event, ids_dict, model, doc_matrix, topK=10):
    unseen_even_tags = jieba.analyse.extract_tags(unseen_event)
    unseen_event_vector = [0] * doc_matrix[1]
    for tag in unseen_even_tags:
        if tag in model.vocabulary_:
            unseen_event_vector[model.vocabulary_[tag]] += 1
    ranking_list = []
    for index, vector in enumerate(doc_matrix):
        score = cosine_similarity([unseen_event_vector], vector)
        ranking_list.append(
            (score[0][0], ids_dict[index])
            )
    ranking_list.sort(reverse=True)
    return list(map(lambda x: x[1], ranking_list[:topK]))

def embedding_propgation(ranking_list, fp=EMBEDDING_FILE):
    with open(EMBEDDING_FILE, 'r') as json_file_in:
        embedding_dict = json.load(json_file_in)
    accumulate_vector = []
    add_count = 0
    for ranking_list_index, id_ in enumerate(ranking_list):
        try:
            added_vector = embedding_dict[id_]
        except KeyError:
            # Due to some events are lack of people book them,
            # they are removed from the training set.
            print("{} is not a significant event so that not included in the training embedding.".format(id_))
            continue
        if add_count == 0:
            accumulate_vector = added_vector
        else:
            for index, (element1, element2) in\
                            enumerate(zip(accumulate_vector, added_vector)):
                accumulate_vector[index] = element1 + element2
        add_count += 1
    print('{} related events.'.format(add_count))
    return list(map(lambda x: x / add_count, accumulate_vector))

def load_unseen(fp=UNSEEN_EVENTS_FILE):
    with open(fp, 'rt') as fin:
        unseen_dict = {}
        for line in fin:
            splitted_line = line.strip().split(',')
            if len(splitted_line) == 1:
                continue
            id_, title = splitted_line
            unseen_dict[id_] = title
        return unseen_dict


if __name__ == "__main__":
    IDS_DICT, TRAINED_MODEL, DOC_MATRIX = vsm()
    UNSEEN_DICT = load_unseen()
    UNSEEN_EMBEDDING_DICT = {}
    for id_, title_string in UNSEEN_DICT.items():
        print('unssenId:', id_)
        ID_LIST =\
            closest_topK(title_string, IDS_DICT, TRAINED_MODEL, DOC_MATRIX)
        UNSEEN_EMBEDDING_DICT[id_] = embedding_propgation(ID_LIST)
        print()
    with open('unssen_events_rep_hpe2.txt', 'wt') as fout:
        fout.write("{}\n".format(len(UNSEEN_EMBEDDING_DICT)))
        for id_, embedding in UNSEEN_EMBEDDING_DICT.items():
            fout.write("{} {}\n".format(id_, ' '.join(map(lambda x:str(round(x, 6)),embedding))))
