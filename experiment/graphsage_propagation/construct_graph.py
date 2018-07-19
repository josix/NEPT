import json
import random
from collections import defaultdict
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np


def load_edges(file_path_list: list) -> list:
    """
    Input: file path.
    Output: The edges of a graph.
    """
    edge_list = []
    for file_path in file_path_list:
        with open(file_path) as fin:
            for line in fin:
                user, *items = line.strip().split()
                for item in items:
                    edge_list.append((f"u{user}", f"{item}"))
    return edge_list


def load_embedding(file_path):
    vertex_embedding = defaultdict(list)
    with open(file_path) as fin:
        fin.readline()
        training_list = []
        for line in fin:
            vertex, *embedding = line.strip().split()
            training_list.append(vertex)
            vertex_embedding[vertex] = list(map(lambda x: float(x), embedding))
        return vertex_embedding, set(training_list)


if __name__ == "__main__":
    EDGES = load_edges(["../../source/training_data_4years.data", "../../source/testing_data_4years.data"])
    EMBEDDING, TRAINING_SET = load_embedding("../../hpe_data/rep.hpe")
    DIM = len(list(EMBEDDING.values())[0])

    G = nx.Graph()
    G.add_edges_from(EDGES)
    VAL_SET = random.sample(TRAINING_SET, int(len(TRAINING_SET)*0.4))
    for node in G.nodes():
        if node in VAL_SET:
            G.node[node]["val"] = True
            G.node[node]["test"] = False
        if node not in TRAINING_SET:
            G.node[node]["val"] = False
            G.node[node]["test"] = True
        else:
            G.node[node]["val"] = False
            G.node[node]["test"] = False
    with open("CompleteTime-G.json", "wt") as fout:
        json.dump(json_graph.node_link_data(G), fout)
    ID_MAP = {}
    CLASS_MAP = {}
    USER_NODE_COUNT = 0
    ITEM_NODE_COUNT = 0
    EMBEDDING_MATRIX = []
    for index, node in enumerate(G.nodes()):
        if node in TRAINING_SET:
            if index == 0:
                EMBEDDING_MATRIX.append(EMBEDDING[node])
            else:
                EMBEDDING_MATRIX.append(EMBEDDING[node])
        else:
            if index == 0:
                EMBEDDING_MATRIX.append([0] * DIM)
            else:
                EMBEDDING_MATRIX.append([0] * DIM)

        if node[0] == "u":
            USER_NODE_COUNT += 1
        else:
            ITEM_NODE_COUNT += 1
        ID_MAP[node] = index
        CLASS_MAP[node] = 0
    with open("CompleteTime-id_map.json", "wt") as fout:
        json.dump(ID_MAP, fout)
    with open("CompleteTime-class_map.json", "wt") as fout:
        json.dump(CLASS_MAP, fout)
    np.save('CompleteTime-feats.npy', np.array(EMBEDDING_MATRIX))
    print(f"#edges: {len(EDGES)}")
    print(f"#nodes: {len(G.nodes())}")
    print(f"#users: {USER_NODE_COUNT}")
    print(f"#items: {ITEM_NODE_COUNT}")
