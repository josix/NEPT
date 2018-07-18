import random

random.seed(11)
item_set = set()
with open('../source/entertainment_transactions_v7_Before20161231.data', 'rt') as fin:
    training_id = []
    for line in fin:
        user, item, title = line.strip().split(',')
        training_id.append(item)
    item_set = item_set | set(training_id)
with open('../source/entertainment_transactions_v7_After20161231.data', 'rt') as fin:
    testing_id = []
    for line in fin:
        user, item, title = line.strip().split(',')
        testing_id.append(item)
    item_set = item_set | set(testing_id)

item_list = list(item_set)
random.shuffle(item_list)
item_num = len(item_list)
query_list = item_list[:int(item_num*0.3)]
for query in query_list:
    print(f" {query}")
