from collections import defaultdict
import random

training_data = defaultdict(list)
testing_data = defaultdict(list)
with open('./temp/kktixItemListAllState.data', 'rt') as fin:
    for line in fin:
        user, *items = line.strip().split()
        items = list(set(items))
        if len(items) < 4:
            continue
        random.shuffle(items)
        cut_index = int((len(items)-1)*0.8)
        training_data[user] = items[:cut_index]
        testing_data[user] = items[cut_index:]

with open('./temp/training_data.data', 'wt') as fout:
    for user in training_data:
        output_str = "{} ".format(user)
        for index, item in enumerate(training_data[user]):
            output_str += "{}\n".format(item) if index == len(training_data[user]) - 1 else "{} ".format(item)
        fout.write(output_str)

with open('./temp/testing_data.data', 'wt') as fout:
    for user in testing_data:
        output_str = "{} ".format(user)
        for index, item in enumerate(testing_data[user]):
            output_str += "{}\n".format(item) if index == len(testing_data[user]) - 1 else "{} ".format(item)
        fout.write(output_str)
