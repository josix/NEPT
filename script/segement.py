#! /usr/bin/python
'''Generate segementation of text by using jieba-zh_TW package.
Repo of jieba-zh_TW: https://github.com/ldkrsi/jieba-zh_TW.git

eg:
    <source.data>                ->             <tags.json>
    eventId    title    otherCol                [{eventId: [tags]},...]
    event1     title1   xxx                     event1: [tag1, tag2, tag3 ....]
    event2     title2   xxx                     event2: [tag1, tag2, tag3 ....]
    ...
'''
from collections import defaultdict
import argparse
import json

import jieba
import jieba.analyse


PARSER = argparse.ArgumentParser()
PARSER.add_argument("file",
                    type=str,
                    help="Relative path of file which is to be converted.")
PARSER.add_argument("-o",
                    "--output",
                    default='../data/tags.json',
                    type=str,
                    help="Output path of converted file.")
ARGS = PARSER.parse_args()
FILEPATH = ARGS.file
OUTPUT = ARGS.output


def event_title_cut(filepath):
    '''Return a dict{event_id: [tags]} '''
    tag_dict = defaultdict(list)
    jieba.set_dictionary("../jieba-zh_TW/jieba/dict.txt")
    with open(filepath, 'rt') as fin:
        for line in fin:
            event_id, event_title, *_ = line.strip().split(',')
            tag_dict[int(event_id)] = \
                jieba.analyse.extract_tags(event_title)
    return tag_dict


if __name__ == "__main__":
    EVENT_TITLE_TAG_DICT = event_title_cut(FILEPATH)
    with open(OUTPUT, 'w', encoding='utf-8') as json_file:
        json.dump(EVENT_TITLE_TAG_DICT,
                  json_file,
                  ensure_ascii=False)
