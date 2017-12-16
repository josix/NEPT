from collections import defaultdict

user_items_list = defaultdict(list)
with open('./temp/etNet_v7col5AllState.train', 'wt') as fout:
    with open('./temp/training_data.data', 'rt') as fin:
        for line in fin:
            user, *items= line.strip().split(' ')
            for item in items:
                fout.write("{} {}\n".format(user, item))
