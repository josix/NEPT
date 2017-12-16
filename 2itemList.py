from collections import defaultdict

user_items_list = defaultdict(list)
with open('../kktix/etNet_v7allState.data', 'rt') as fin:
    for line in fin:
        user, item, *others = line.strip().split(' ')
        user_items_list[user].append(item)

with open('./temp/kktixItemListAllState.data', 'wt') as fout:
    for user in user_items_list:
        output_str = "{} ".format(user)
        for index, item in enumerate(user_items_list[user]):
            if index == len(user_items_list[user]) - 1:
                output_str += "{}\n".format(item)
            else:
                output_str += "{} ".format(item)
        fout.write(output_str)
