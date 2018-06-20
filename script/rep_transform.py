"""
Convert original representation txt file into the json format
(excludes user entity)
"""
import argparse
import json

PARSER = argparse.ArgumentParser()
PARSER.add_argument("file",
                    type=str,
                    help="Relative path of the representation file which is to be converted.")
PARSER.add_argument("-o",
                    "--output",
                    default='./rep.json',
                    type=str,
                    help="Output path of converted file. (default: rep.json)")
ARGS = PARSER.parse_args()
FILEPATH = ARGS.file
OUTPUT = ARGS.output

ENTITY_REP_MAPPING = {}
with open(FILEPATH, 'rt') as fin:
    fin.readline()
    LINES = fin.readlines()
    for line in LINES:
        entity_id, *rep = line.strip().split()
        if entity_id[0] == 'u':
            continue
        ENTITY_REP_MAPPING[entity_id] = rep

with open(OUTPUT, 'w') as outfile:
    json.dump(ENTITY_REP_MAPPING, outfile)
