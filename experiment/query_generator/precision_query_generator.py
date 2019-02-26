import argparse
import random

PARSER = argparse.ArgumentParser()
PARSER.add_argument("user_to_items",
                    type=str,
                    help="User-items data")
PARSER.add_argument("popular_events",
                    type=str,
                    help="top k popular events data")

ARGS = PARSER.parse_args()
FILEPATH = ARGS.user_to_items
POP_FILEPATH = ARGS.popular_events
random.seed(11)
def load_popular_event() -> set:
    with open(POP_FILEPATH) as fin:
        return {line.strip().split(',')[0] for line in fin}

popular_events = load_popular_event()
with open(FILEPATH, 'rt') as fin:
    for line in fin:
        user, *item_list = line.strip().split(' ')
        item_num = len(item_list)
        if item_num < 10: # for measuring precision@5 or strong user
            continue
        # print(user)
        # print(item_list)
        random.shuffle(item_list)
        # query_list = item_list[:int(item_num*0.5)] # multi queries for one user (came from 50% of previous record)
        try:
            item_list = [random.choice([item for item in item_list if item in popular_events])]
        except IndexError:
            continue
        query_list = [item_list[0]] # one query for one user
        for query in query_list:
            print(f"{user} {query}")
