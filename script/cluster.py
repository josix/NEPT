import argparse
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np

def load_word_mapping(fp):
    """
    load word_mapping:
    w1, apple => Dict['w1', 'apple'] 
    """
    with open(fp) as fin:
        return {line.strip().split(',')[0]:line.strip().split(',')[1] for line in fin}

def load_semantic_emb(fp, word_id_to_word):
    id_to_emb = {}
    index_to_word = {}
    with open(fp, 'rt') as fin:
        fin.readline()
        index = 0
        for line in fin:
            if line[0] != 'w':
                continue
            word_id, *emb = line.strip().split()
            id_to_emb[word_id] = [float(value) for value in emb]
            index_to_word[index] = word_id_to_word[word_id]
            index += 1
    return id_to_emb, index_to_word

def train_cluster(training_data):
    kmeans = KMeans(
            n_clusters=5000,
            random_state=0,
            max_iter=30000,
            n_jobs=20,
            verbose=1).fit(training_data)
    return kmeans


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("rep",
            type=str,
            help="The path of semantic embedding file")
    PARSER.add_argument("word_mapping_file",
            type=str,
            help="The mapping file between the words and ids")
    PARSER.add_argument("user_word_graph",
            type=str,
            help="The original user-word graph")
    PARSER.add_argument("-o",
            "--output",
            default="./user-cluster.data",
            type=str,
            help="Output path of bipartite graph. (default: user-cluster.data)")
    ARGS = PARSER.parse_args()

    WORD_ID_TO_WORD = load_word_mapping(ARGS.word_mapping_file)
    WORD_ID_TO_EMB, INDEX_TO_WORD = load_semantic_emb(ARGS.rep, WORD_ID_TO_WORD)
    data = np.array([value for value in WORD_ID_TO_EMB.values()])
    print("Training data size:", len(data), len(data[0]))
    model = train_cluster(data)
    with open('textrank_mapping.txt', 'wt') as fout:
        for index, label in enumerate(model.labels_):
            fout.write("cluster{},{}\n".format(label, INDEX_TO_WORD[index]))

    WORD_TO_INDEX = {value: key for key, value in INDEX_TO_WORD.items()}
    USER_CLUSTER_TO_WEIGHT_LIST = defaultdict(list)
    with open(ARGS.user_word_graph, 'rt') as fin:
        for line in fin:
            user, item, weight = line.strip().split()
            USER_CLUSTER_TO_WEIGHT_LIST[(user, model.labels_[WORD_TO_INDEX[WORD_ID_TO_WORD[item]]])].append(float(weight))

    USER_CLUSTER_TO_WEIGHT = {}
    for key in USER_CLUSTER_TO_WEIGHT_LIST:
        USER_CLUSTER_TO_WEIGHT[key] = sum(USER_CLUSTER_TO_WEIGHT_LIST[key])/len(USER_CLUSTER_TO_WEIGHT_LIST[key])

    with open(ARGS.output, 'wt') as fout:
        for (user, cluster_index), weight in USER_CLUSTER_TO_WEIGHT.items():
            fout.write("{} cluster{} {}\n".format(user, model.labels_[cluster_index], weight))

