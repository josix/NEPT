import random

random.seed(11)
with open('../data/test.data', 'rt') as fin:
    for line in fin:
        user, *item_list = line.strip().split(' ')
        item_num = len(item_list)
        # print(user)
        # print(item_list)
        random.shuffle(item_list)
        query_list = item_list[:int(item_num*0.5)]
        for query in query_list:
            print(f"u{user} {query}")
