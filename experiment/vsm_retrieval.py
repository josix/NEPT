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
PARSER.add_argument("embedding_file",
                    type=str,
                    help="The embedding json file")
PARSER.add_argument("corpus_file",
                    type=str,
                    help="The items' title text file (json) for training vsm")
ARGS = PARSER.parse_args()
EMBEDDING_FILE = ARGS.embedding_file
CORPUS_FILE = ARGS.corpus_file

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
    jieba.set_dictionary("./jieba-zh_TW/jieba/dict.txt")
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
        if ranking_list_index == 0:
            accumulate_vector = added_vector
        else:
            for index, (element1, element2) in\
                            enumerate(zip(accumulate_vector, added_vector)):
                accumulate_vector[index] = element1 + element2
        add_count += 1
    print(add_count)
    return list(map(lambda x: x / add_count, accumulate_vector))


if __name__ == "__main__":
    IDS_DICT, TRAINED_MODEL, DOC_MATRIX = vsm()
    while True:
        input_str = input("Enter event title:")
        ID_LIST = closest_topK(input_str, IDS_DICT, TRAINED_MODEL, DOC_MATRIX)
        print(embedding_propgation(ID_LIST))

