import argparse
import json
from networkx.readwrite import json_graph
import numpy as np

PARSER = argparse.ArgumentParser()
PARSER.add_argument("graph_file",
                    type=str,
                    help="Graph file path.")
PARSER.add_argument("embedding_file",
                    type=str,
                    help="Embedding file came from GraphSAGE Embedding Propagation.")
PARSER.add_argument("mapping_file",
                    type=str,
                    help="File mapping embedding to id")
ARGS = PARSER.parse_args()
GRAPH_FP = ARGS.graph_file
EMBED_FP = ARGS.embedding_file
MAPPING_FP = ARGS.mapping_file

EMBEDDING = np.load(EMBED_FP)
G = json_graph.node_link_graph(json.load(open(GRAPH_FP)))

TRAIN_IDS = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
TEST_IDS = [n for n in G.nodes() if G.node[n]['test']]
ID_MAP = {}
with open(MAPPING_FP) as fin:
    for index, line in enumerate(fin):
        ID_MAP[line.strip()] = index

embedding_mapping = {}
for id_ in TRAIN_IDS:
    embedding_mapping[id_] = EMBEDDING[ID_MAP[id_]].tolist()
for id_ in TEST_IDS:
    embedding_mapping[id_] = EMBEDDING[ID_MAP[id_]].tolist()

with open('rep.graphsage', 'wt') as fout:
    fout.write(f"{len(embedding_mapping)} {len(EMBEDDING[0])}\n")
    for vertex, embedding in embedding_mapping.items():
        fout.write(f"{vertex} ")
        fout.write("{}\n".format(' '.join(map(str, embedding))))
