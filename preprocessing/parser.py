import argparse
from bs4 import BeautifulSoup

PARSER = argparse.ArgumentParser()
PARSER.add_argument("file",
                    type=str,
                    help="Relative path of file which is to be converted.")
PARSER.add_argument("--testing",
                    type=bool,
                    default=False)
ARGS = PARSER.parse_args()
FILEPATH = ARGS.file
with open(FILEPATH, 'rt') as fin:
    seen_set = set()
    for line in fin:
        if ARGS.testing and "2018" not in line:
            continue
        line = line.split(',')
        eventId = line[0]
        if eventId in seen_set:
            continue
        seen_set.add(eventId)
        broken_html = line[5]
        # print(line.split(','))
        title = line[4]
        bsObj = BeautifulSoup(broken_html, 'lxml')
        tags = {tag.name for tag in bsObj.find_all() if tag.name != 'br'}
        #print(tags, broken_html)
        print(
            f"{eventId},{title},{bsObj.find_all('html')[0].text if bsObj.find_all('html') != [] else ''}"
        )
        #print(bsObj.find_all('html')[0].text)
