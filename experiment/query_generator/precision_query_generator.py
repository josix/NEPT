import random

random.seed(11)
with open('../../hpe_data/test.data', 'rt') as fin:
    for line in fin:
        user, *item_list = line.strip().split(' ')
        item_num = len(item_list)
        if item_num < 10: # for measuring precision@10
            continue
        # print(user)
        # print(item_list)
        random.shuffle(item_list)
        # query_list = item_list[:int(item_num*0.5)]
        query_list = [item_list[0]]
        for query in query_list:
            print(f"u{user} {query}")
