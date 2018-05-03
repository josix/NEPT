"""
This module will convert user-item pairs into a row of user and
its corresponding list of items.
eg:
    <source.data>                ->              <itemsList.data>
    user    item    otherCol                     user  itemList
    u1      item1   xxx                          u1    item1 item2 item3 ....
    u1      item2   xxx
    u1      item3   xxx
    ...
"""
from collections import defaultdict
import argparse

PARSER = argparse.ArgumentParser()
PARSER.add_argument("file",
                    type=str,
                    help="Relative path of file which is to be converted.")
PARSER.add_argument("-o",
                    "--output",
                    default='./itemsList.data',
                    type=str,
                    help="Output path of converted file.")
ARGS = PARSER.parse_args()
FILEPATH = ARGS.file
OUTPUT = ARGS.output

USER_ITEMS_LIST = defaultdict(list)
with open(FILEPATH, 'rt') as fin:
    for line in fin:
        user, item, *others = line.strip().split(' ')
        USER_ITEMS_LIST[user].append(item)

with open(OUTPUT, 'wt') as fout:
    for user in USER_ITEMS_LIST:
        output_str = "{} ".format(user)
        for index, item in enumerate(USER_ITEMS_LIST[user]):
            if index == len(USER_ITEMS_LIST[user]) - 1:
                output_str += "{}\n".format(item)
            else:
                output_str += "{} ".format(item)
        fout.write(output_str)
