import random

random.seed(11)
def load_popular_event() -> set:
    with open('../../source/query_source/popular_events_100.txt') as fin:
        return {line.strip().split(',')[0] for line in fin}

popular_events = load_popular_event()
with open('../../log_transaction_data/user_to_items_before2018.data', 'rt') as fin:
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
