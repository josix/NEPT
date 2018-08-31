#! /usr/bin/python
'''Generate the textrank of words by using jieba-zh_TW package.
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

from jieba import analyse
import jieba
from tqdm import tqdm

PARSER = argparse.ArgumentParser()
PARSER.add_argument("file",
                    type=str,
                    help="Relative path of file which is to be converted.")
PARSER.add_argument("-o",
                    "--output",
                    default='./data/textrank',
                    type=str,
                    help="Output the prefix of converted file.")
ARGS = PARSER.parse_args()
FILEPATH = ARGS.file
OUTPUT = ARGS.output

def event_title_cut(filepath: str) -> dict:
    '''Return a dict{event_id: [words]} '''
    tag_dict = dict()
    jieba.set_dictionary("./jieba-zh_TW/jieba/dict.txt")
    with open(filepath, 'rt') as fin:
        for line in tqdm(fin):
            event_id, event_title, *event_description_list = line.strip().split(',')
            event_description = " ".join(event_description_list)
            title_tags = [(word, 1.0) for word in jieba.analyse.extract_tags(event_title)]
            tag_dict[int(event_id)] = [
                    *title_tags,
                    *jieba.analyse.textrank(event_description, topK=10, withWeight=True, allowPOS=('ns', 'n'))
                    ]
    return tag_dict
if __name__ == "__main__":
    EVENT_TITLE_TAG_DICT = event_title_cut(FILEPATH)
    with open(OUTPUT+".json", 'w', encoding='utf-8') as json_file:
        json.dump(EVENT_TITLE_TAG_DICT,
                  json_file,
                  ensure_ascii=False)
    with open(OUTPUT+".txt", 'w', encoding="utf-8") as fout:
        for key, value in EVENT_TITLE_TAG_DICT.items():
            fout.write(f"{key}, {'/ '.join([x[0] for x in value])}\n")
    with open(OUTPUT+"_mapping.txt", 'w') as fout:
        corpus = set([word[0] for word_list in EVENT_TITLE_TAG_DICT.values() for word in word_list])
        for index, word in enumerate(corpus):
            fout.write('w{},{}\n'.format(index, word))
