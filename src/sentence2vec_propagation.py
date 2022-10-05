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
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.models import KeyedVectors
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
MAX_EPOCHS = 10
SIZE = 64
def sentence2vec(fp=CORPUS_FILE):
    with open(fp, 'r') as json_file_in:
        item_tags_dict = json.load(json_file_in)
        corpus = [
            TaggedDocument(words=value, tags=[id_key])
            for id_key, value in item_tags_dict.items()
        ]

        model = Doc2Vec(min_count=0,vector_size=SIZE,window=1,dbow_words=0,dm=1, dm_concat=1, alpha=0.025, min_alpha=0.025, epochs=MAX_EPOCHS)
        model.build_vocab(corpus)
        model.train(corpus, epochs=model.iter, total_examples=model.corpus_count)
        annoy_index = AnnoyIndex(SIZE)
        title_vec = {}
        for id_ in item_tags_dict.keys():
            title_vec[id_] = model.infer_vector(item_tags_dict[id_])
            annoy_index.add_item(int(id_), title_vec[id_])
        annoy_index.build(10) # 10 trees
        annoy_index.save('sentence2vec.ann')
        return model, title_vec

def closest_topK(unseen_event, model, dim, topK=10):
    unseen_even_tags = jieba.analyse.extract_tags(unseen_event)
    unseen_event_vector = model.infer_vector(unseen_even_tags)
    annoy_index = AnnoyIndex(dim)
    annoy_index.load('sentence2vec.ann')
    ranking_list = annoy_index.get_nns_by_vector(unseen_event_vector, 10, search_k=-1, include_distances=True)
    return list(zip(ranking_list[0], ranking_list[1]))

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
            print(
                f"{id_} is not a significant event so that not included in the training embedding."
            )

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
    print(
        f'weight list: {list(map(lambda x: x / accumulate_weight, weight_list))}'
    )

    print(f'{add_count} related events.')
    return list(map(lambda x: x / accumulate_weight, accumulate_vector))

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
    SENTENCE2VEC_MODEL, TITLE_SENTENCE2VEC = sentence2vec()
    UNSEEN_DICT = load_unseen()
    UNSEEN_EMBEDDING_DICT = {}
    for id_, title_string in UNSEEN_DICT.items():
        print('unssenId:', id_)
        ID_LIST =\
            closest_topK(title_string, SENTENCE2VEC_MODEL, SIZE)
        print(ID_LIST)
        UNSEEN_EMBEDDING_DICT[id_] = embedding_propgation(ID_LIST, weight_func = lambda x : 1 / (0.00001 + x))
        print()
    with open('unssen_events_rep_hpe(sentence2vec_weight_angular_size_1_dm_iter10).txt', 'wt') as fout:
        fout.write(f"{len(UNSEEN_EMBEDDING_DICT)}\n")
        for id_, embedding in UNSEEN_EMBEDDING_DICT.items():
            fout.write(f"{id_} {' '.join(map(lambda x: str(round(x, 6)), embedding))}\n")
