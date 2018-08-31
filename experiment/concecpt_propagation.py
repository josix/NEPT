"""
Using the vsm model to pass the embedding from the high similarity items entity
to the unseen item entity.
"""
import json
import argparse
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
PARSER.add_argument("concept_folder",
                    type=str,
                    help="The concepts data for all the events")
ARGS = PARSER.parse_args()
UNSEEN_EVENTS_FILE = ARGS.unseen_event_file
EMBEDDING_FILE = ARGS.embedding_file
CORPUS_FILE = ARGS.corpus_file
CONCEPT_FOLDER = ARGS.concept_folder
jieba.set_dictionary("./jieba-zh_TW/jieba/dict.txt")
MAX_EPOCHS = 10
SIZE = 128
def concept_combine(concept_embedding, concept_mapping, fp=CORPUS_FILE):
    with open(fp, 'r') as json_file_in:
        item_tags_dict = json.load(json_file_in)
        corpus = []
        event_vec = {}
        annoy_index = AnnoyIndex(SIZE)
        for id_key, words in item_tags_dict.items():
            event_concept_embeddings = []
            for word, weight in words:
                try:
                    event_concept_embeddings.append(concept_embedding[concept_mapping[word]])
                except KeyError:
                    continue
                if event_concept_embeddings == []:
                    continue
                event_vec[id_key] = [sum(value) / len(value) for value in  zip(*event_concept_embeddings)]
                annoy_index.add_item(int(id_key), event_vec[id_key])
        annoy_index.build(10) # 10 trees
        annoy_index.save('cc2vec.ann')
        return event_vec

def closest_topK(unseen_event, concept_embedding, concept_mapping, dim, topK=10):
    """
    unseen_event: (title: str, description: str)
    concept_embedding: {word_id : [emb]}
    concept_mapping: {word_id : word_string}
    """
    unseen_event_title_tags = jieba.analyse.extract_tags(unseen_event[0])
    unseen_event_description_words = jieba.analyse.textrank(unseen_event[1], topK=10, withWeight=False, allowPOS=('ns', 'n'))
    print('title words:', unseen_event_title_tags)
    print('description words:', unseen_event_description_words)
    event_concept_embeddings = []
    for word in [*unseen_event_title_tags, *unseen_event_description_words]:
        try:
            event_concept_embeddings.append(concept_embedding[concept_mapping[word]])
        except KeyError:
            continue
    unseen_event_vector = [ sum(value) / len(value) for value in  zip(*event_concept_embeddings)]
    if unseen_event_vector == []:
        unseen_event_vector = [0] * dim
    annoy_index = AnnoyIndex(dim)
    annoy_index.load('cc2vec.ann')
    ranking_list = annoy_index.get_nns_by_vector(unseen_event_vector, 10, search_k=-1, include_distances=True)
    propgation_list = []
    for id_, score in zip(ranking_list[0], ranking_list[1]):
        propgation_list.append((id_, score))
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
            added_vector = embedding_dict[str(id_)]
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
            unseen_dict[id_] = title, description
        return unseen_dict

def load_concept(fp=CONCEPT_FOLDER):
    embedding = {}
    with open(CONCEPT_FOLDER + '/rep_textrank_weight.line2') as fin:
        fin.readline()
        for line in fin:
            id_, *vector = line.strip().split()
            embedding[id_] = [ float(value) for value in vector]
    word_id_mapping = {}
    with open(CONCEPT_FOLDER + '/textrank_mapping.txt') as fin:
        for line in fin:
            word_id, word = line.strip().split(',')
            word_id_mapping[word] = word_id
    return embedding, word_id_mapping

if __name__ == "__main__":
    CONCEPT_EMBEDDING, CONCEPT_ID_MAPPING = load_concept()
    TITLE_CC2VEC = concept_combine(CONCEPT_EMBEDDING, CONCEPT_ID_MAPPING)
    UNSEEN_DICT = load_unseen()
    UNSEEN_EMBEDDING_DICT = {}
    for id_, content in UNSEEN_DICT.items():
        print('unssenId:', id_)
        ID_LIST =\
            closest_topK(content, CONCEPT_EMBEDDING, CONCEPT_ID_MAPPING, SIZE)
        print(ID_LIST)
        UNSEEN_EMBEDDING_DICT[id_] = embedding_propgation(ID_LIST, weight_func=lambda x: 1 / (0.00001 + x))
        print()
    with open('unssen_events_rep_hpe(cc2vec_weight_angular_description).txt', 'wt') as fout:
        fout.write("{}\n".format(len(UNSEEN_EMBEDDING_DICT)))
        for id_, embedding in UNSEEN_EMBEDDING_DICT.items():
            fout.write("{} {}\n".format(id_, ' '.join(map(lambda x:str(round(x, 6)),embedding))))
