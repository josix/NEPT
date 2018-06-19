"""
This module will split source data into training set and testing set.
eg:
    <source.data>           ->       <training.datat>     and    <testing.data>
    user   itemList                  user itemList              user itemList
    u1     item1 item2 item3 ...     u1   item1 item3 item4     u1 item2  item5

"""
from collections import defaultdict
import random
import argparse

PARSER = argparse.ArgumentParser()
PARSER.add_argument("file",
                    type=str,
                    help="Relative path of file which is to be converted.")
PARSER.add_argument("-o1",
                    "--train_output",
                    default='./training.data',
                    type=str,
                    help="Output path of training file. Default to training.data.")
PARSER.add_argument("-o2",
                    "--test_output",
                    default='./testing.data',
                    type=str,
                    help="Output path of testing file. Default to testing.data")
ARGS = PARSER.parse_args()
FILEPATH = ARGS.file
TRAIN_OUTPUT = ARGS.train_output
TEST_OUTPUT = ARGS.test_output

TRAINING_DATA = defaultdict(list)
TESTING_DATA = defaultdict(list)
with open(FILEPATH, 'rt') as fin:
    for line in fin:
        user, *items = line.strip().split()
        items = list(set(items))
        if len(items) < 4:  # Drop the users who have less using experience
            continue
        random.shuffle(items)
        cut_index = int((len(items)-1)*0.8)
        TRAINING_DATA[user] = items[:cut_index]
        TESTING_DATA[user] = items[cut_index:]

with open(TRAIN_OUTPUT, 'wt') as fout:
    for user in TRAINING_DATA:
        output_str = "{} ".format(user)
        for index, item in enumerate(TRAINING_DATA[user]):
            output_str +=\
                "{}\n".format(item) if index == len(TRAINING_DATA[user]) - 1\
                else "{} ".format(item)
        fout.write(output_str)

with open(TEST_OUTPUT, 'wt') as fout:
    for user in TESTING_DATA:
        output_str = "{} ".format(user)
        for index, item in enumerate(TESTING_DATA[user]):
            output_str +=\
                   "{}\n".format(item) if index == len(TESTING_DATA[user]) - 1\
                   else "{} ".format(item)
        fout.write(output_str)
