"""
Using the vsm model to pass the embedding from the high similarity items entity
to the unseen item entity.
"""
import json
import argparse
import pickle as pickle
import sys
sys.path.insert(0, "./jieba-zh_TW_NEPT_src")

import jieba
import jieba.analyse
from jieba import analyse
from annoy import AnnoyIndex
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity


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
PARSER.add_argument("--textrank_word2vec",
                    type=bool,
                    default=False)
PARSER.add_argument("--textrank_idf",
                    type=bool,
                    default=False)
PARSER.add_argument("--embedrank",
                    type=bool,
                    default=False)
ARGS = PARSER.parse_args()
UNSEEN_EVENTS_FILE = ARGS.unseen_event_file
EMBEDDING_FILE = ARGS.embedding_file
CORPUS_FILE = ARGS.corpus_file
CONCEPT_FOLDER = ARGS.concept_folder
jieba.set_dictionary("./jieba-zh_TW_NEPT_src/jieba/dict.txt")
MAX_EPOCHS = 10
SIZE = 128
# Switch embedrank, textrank_w2v, textrank_vsm
MODEL=None
if ARGS.embedrank:
    try:
        print('load doc2vec model')
        MODEL = Doc2Vec.load(CONCEPT_FOLDER+"/doc2vec.model")
    except FileNotFoundError:
        pass
elif ARGS.textrank_word2vec:
    try:
        print('load word2vec model')
        MODEL = word2vec.Word2Vec.load(CONCEPT_FOLDER+"/word2vec.model")
    except FileNotFoundError:
        pass
elif ARGS.textrank_idf:
    try:
        print('load vsm ')
        MODEL = pickle.load(open(CONCEPT_FOLDER+"/vsm_model.pickle", 'rb'))
    except FileNotFoundError:
        pass

def gen_event_lbl_emb(concept_embedding, concept_mapping, fp=CORPUS_FILE):
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
                # call spotify annoy
                annoy_index.add_item(int(id_key), event_vec[id_key])
        # For K-nearest neighbor retrieval
        annoy_index.build(10) # 10 trees
        annoy_index.save('cc2vec.ann')
        return event_vec

def textrank_getkeywords(paragraph):
    if not MODEL:
        return jieba.analyse.textrank(paragraph, topK=10, withWeight=False, allowPOS=('ns', 'n'))
    else:
        # Switch word2vec or idf
        if ARGS.textrank_idf:
            return jieba.analyse.textrank_vsm(paragraph, topK=10, withWeight=False, allowPOS=('ns', 'n'), vsm=MODEL)
        elif ARGS.textrank_word2vec:
            return jieba.analyse.textrank_similarity(paragraph, topK=10, withWeight=False, allowPOS=('ns', 'n'), word_embedding=MODEL)

def embedrank_getkeywords(paragraph, withWeight=False):
    '''Return a list[(word, weight)] or list[word] '''
    textrank = analyse.TextRank()
    textrank.pos_filt = frozenset(('ns', 'n'))
    words = set()
    for word_pair in textrank.tokenizer.cut(paragraph):
        if textrank.pairfilter(word_pair):
            words.add(word_pair.word)
    doc_vec = MODEL.infer_vector(words)
    candidate_keywords = []
    for word in words:
        word_vec = MODEL.infer_vector(word)
        candidate_keywords.append((word, float(cosine_similarity([word_vec], [doc_vec])[0][0])))
    candidate_keyword = sorted(candidate_keywords, key=lambda x: x[1], reverse=True)
    if withWeight:
        return candidate_keywords[:10]
    else:
        return [word for word, weight in candidate_keyword[:10]]

def closest_topK(unseen_event, concept_embedding, concept_mapping, dim, topK=10):
    """
    unseen_event: (title: str, description: str)
    concept_embedding: {word_id : [emb]}
    concept_mapping: {word_id : word_string}
    """
    unseen_event_title_tags = jieba.analyse.extract_tags(unseen_event[0])

    # Switch textrank or embedrank
    if ARGS.embedrank:
        unseen_event_description_words = embedrank_getkeywords(unseen_event[1])
    else:
        unseen_event_description_words = textrank_getkeywords(unseen_event[1])

    print('title words:', unseen_event_title_tags)
    print('description words:', unseen_event_description_words)
    # Generate the label embedding for a new item
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
    # Find topK colest item according to the label embedding
    ranking_list = annoy_index.get_nns_by_vector(unseen_event_vector, 10, search_k=-1, include_distances=True)
    propgation_list = []
    for id_, score in zip(ranking_list[0], ranking_list[1]):
        propgation_list.append((id_, score))
    return propgation_list

def embedding_propgation(ranking_list, id_to_emb, weight_func = lambda x : 1):
    accumulate_vector = []
    accumulate_weight = 0
    weight_list = []
    add_count = 0
    for ranking_list_index, (id_, score) in enumerate(ranking_list):
        try:
            added_vector = id_to_emb[str(id_)]
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
    with open(CONCEPT_FOLDER + '/rep.line2') as fin:
        fin.readline()
        for line in fin:
            id_, *vector = line.strip().split()
            embedding[id_] = [ float(value) for value in vector]
    word_id_mapping = {}
    file_name = "/embedrank_mapping.txt" if ARGS.embedrank else "/textrank_mapping.txt" # Switch textrank or embedrank
    with open(CONCEPT_FOLDER + file_name) as fin:
        for line in fin:
            word_id, word = line.strip().split(',')
            word_id_mapping[word] = word_id
    return embedding, word_id_mapping

if __name__ == "__main__":
    CONCEPT_EMBEDDING, CONCEPT_ID_MAPPING = load_concept()
    line_event_to_label_emb = gen_event_lbl_emb(CONCEPT_EMBEDDING, CONCEPT_ID_MAPPING)
    UNSEEN_DICT = load_unseen()
    UNSEEN_EMBEDDING_DICT = {}
    with open(EMBEDDING_FILE, 'r') as json_file_in:
        hpe_event_to_item_emb = json.load(json_file_in)
    for id_, content in UNSEEN_DICT.items():
        print('unssenId:', id_)
        ID_LIST =\
            closest_topK(content, CONCEPT_EMBEDDING, CONCEPT_ID_MAPPING, SIZE)
        print(ID_LIST)
        UNSEEN_EMBEDDING_DICT[id_] = embedding_propgation(ID_LIST, line_event_to_label_emb, weight_func=lambda x: 1 / (0.00001 + x)) # params to trained
        print()
    with open('unseen_events_label_embedding(texrank).txt', 'wt') as fout:
        fout.write("{}\n".format(len(UNSEEN_EMBEDDING_DICT)))
        for id_, embedding in UNSEEN_EMBEDDING_DICT.items():
            fout.write("{} {}\n".format(id_, ' '.join(map(lambda x:str(round(x, 6)),embedding))))
