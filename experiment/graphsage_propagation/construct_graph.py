import json
import random
import networkx as nx
from networkx.readwrite import json_graph


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


if __name__ == "__main__":
    EDGES = load_edges(["../../source/training_data_4years.data", "../../source/testing_data_4years.data"])
    G = nx.Graph()
    G.add_edges_from(EDGES)
    NODE_SAMPLING = random.sample(G.nodes(), int(len(G.nodes())*0.4))
    VAL_SET = NODE_SAMPLING[:int(len(NODE_SAMPLING)/2)]
    TEST_SET = NODE_SAMPLING[int(len(NODE_SAMPLING)/2)+1:]
    for node in G.nodes():
        if node in VAL_SET:
            G.node[node]["val"] = True
            G.node[node]["test"] = False
        elif node in TEST_SET:
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
    for index, node in enumerate(G.nodes()):
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
    print(f"#edgs: {len(EDGES)}")
    print(f"#nodes: {len(G.nodes())}")
    print(f"#users: {USER_NODE_COUNT}")
    print(f"#items: {ITEM_NODE_COUNT}")
