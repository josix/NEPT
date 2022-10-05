import json
import csv
import argparse

PARSER = argparse.ArgumentParser()
PARSER.add_argument("user_log",
                    type=str,
                    help="The user booking log")
PARSER.add_argument("event_keyword_json",
                    type=str,
                    help="The event textrank json file")
PARSER.add_argument("word_mapping_file",
                    type=str,
                    help="The mapping file between the words and ids")
PARSER.add_argument("-o",
                    "--output",
                    default='../data/user-label.data',
                    type=str,
                    help="Output path of converted file. (default: user-label.data)")
ARGS = PARSER.parse_args()
USER_LOG = ARGS.user_log
EVENT_KEYWORD_JSON = ARGS.event_keyword_json
WORD_MAPPING = ARGS.word_mapping_file
OUTPUT = ARGS.output

if __name__ == "__main__":
    event_keywords = json.load(open(EVENT_KEYWORD_JSON))
    word_id_mapping = {}
    with open(WORD_MAPPING) as fin:
        reader = csv.reader(fin, skipinitialspace=True, quotechar="'")
        for row in reader:
            word_id_mapping[row[1]]=row[0]

    with open(OUTPUT, 'wt') as fout:
        with open(USER_LOG, 'rt') as fin:
            for line in fin:
                user, item, user_item_weight = line.strip().split(' ')
                try:
                    for word, textrank_weight in event_keywords[item]:
                        # on item-word graph
                        # weight = textrank_weight
                        # fout.write('{} {} {}\n'.format(item, word_id_mapping[word], weight))
                        # on user-word graph
                        weight = textrank_weight * int(user_item_weight)
                        fout.write(f'{user} {word_id_mapping[word]} {weight}\n')
                except KeyError:
                    continue
