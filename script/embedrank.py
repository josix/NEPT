#! /usr/bin/python
'''Generate the embedrank of words by using jieba-zh_TW package and doc2vec model.
Repo of jieba-zh_TW: https://github.com/ldkrsi/jieba-zh_TW.git

eg:
    <source.data>                ->             <tags.json>
    eventId    title    otherCol                [{eventId: [word]},...]
    event1     title1   xxx                     event1: [word1, word2, word3 ....]
    event2     title2   xxx                     event2: [word1, word2, word3 ....]
    ...
'''
import argparse
import json
import os.path

from jieba import analyse
import jieba
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.models import KeyedVectors

PARSER = argparse.ArgumentParser()
PARSER.add_argument("file",
                    type=str,
                    help="Relative path of file which is to be converted.")
PARSER.add_argument("-o",
                    "--output",
                    default='./data/embedrank',
                    type=str,
                    help="Output the prefix of converted file.")
PARSER.add_argument("-l",
                    "--load",
                    default='./data/embedrank',
                    type=str,
                    help="Load pretraind doc2vec and doc_words_mapping")
ARGS = PARSER.parse_args()
FILEPATH = ARGS.file
LOAD = ARGS.load
OUTPUT = ARGS.output

MAX_EPOCHS = 10
SIZE = 64
def doc2vec_train(filepath: str):
    '''
    Return a Doc2VecKeyedVectors
    '''
    with open(filepath, 'rt') as fin:
        textrank = analyse.TextRank()
        textrank.pos_filt = frozenset(('ns', 'n'))
        jieba.set_dictionary("./jieba-zh_TW/jieba/dict.txt")
        paragraphs = []
        document_to_words = {}
        for line in tqdm(fin):
            event_id, *event_description_list = line.strip().split(',')
            event_description = " ".join(event_description_list)
            document = []
            for word_pair in textrank.tokenizer.cut(event_description):
                if textrank.pairfilter(word_pair):
                    document.append(word_pair.word)
            document_to_words[event_id] = list(set(document))
            paragraphs.append(TaggedDocument(words=document, tags=[event_id]))

        model = Doc2Vec(
                min_count=0,
                vector_size=SIZE,
                window=3,
                dbow_words=0,
                dm=1,
                dm_concat=1,
                alpha=0.025,
                min_alpha=0.025,
                worder=4,
                epochs=MAX_EPOCHS)
        model.build_vocab(paragraphs)
        model.train(paragraphs, epochs=model.iter, total_examples=model.corpus_count)
        model.save(OUTPUT + '/doc2vec.model')
        with open(OUTPUT + '/doc_words_training.json', 'wt') as json_out:
            json.dump(document_to_words, json_out, ensure_ascii=False)
        return model, document_to_words

def get_keywords(model, words: set) -> list:
    '''Return a list[(word, weight)] '''
    doc_vec = model.infer_vector(words)
    candidate_keywords = []
    for word in words:
        word_vec = model.infer_vector(word)
        candidate_keywords.append((word, float(cosine_similarity([word_vec], [doc_vec])[0][0])))
    candidate_keyword = sorted(candidate_keywords, key=lambda x: x[1], reverse=True)
    return candidate_keywords[:10]

if __name__ == "__main__":
    # check model exists
    DOC2VEC_MODEL = None
    doc_to_words = None
    if LOAD and os.path.exists(LOAD+'/doc2vec.model'):
        DOC2VEC_MODEL = Doc2Vec.load(LOAD+'/doc2vec.model')
    if LOAD and os.path.exists(LOAD+'/doc_words_training.json'):
        with open(LOAD+'/doc_words_training.json') as json_in:
            doc_to_words = json.load(json_in)
    if DOC2VEC_MODEL is None or doc_to_words is None:
        DOC2VEC_MODEL, doc_to_words = doc2vec_train(FILEPATH)

    title_to_words = {}
    with open(FILEPATH, 'rt') as fin:
        for line in fin:
            event_id, event_title, *_ = line.strip().split(',')
            title_to_words[event_id] = [(word, 1.0) for word in jieba.analyse.extract_tags(event_title)]

    doc_to_keywords = {}
    for doc_id, words in doc_to_words.items():
        doc_to_keywords[doc_id] = [*title_to_words[doc_id], *get_keywords(DOC2VEC_MODEL, words)]

    with open(OUTPUT+"/embedrank.json", 'w', encoding='utf-8') as json_file:
        json.dump(doc_to_keywords,
                  json_file,
                  ensure_ascii=False)
    with open(OUTPUT+"/embedrank.txt", 'w', encoding="utf-8") as fout:
        for key, value in doc_to_keywords.items():
            fout.write(f"{key}, {'/ '.join([x[0] for x in value])}\n")
    with open(OUTPUT+"/embedrank_mapping.txt", 'w') as fout:
        corpus = set([word[0] for word_list in doc_to_keywords.values() for word in word_list])
        for index, word in enumerate(corpus):
            fout.write('w{},{}\n'.format(index, word))
