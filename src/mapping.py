"""
This script trains a mapping matrix that can map each embedding in sematic space
into corresponded embedding in preference space.
"""
import argparse

import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential

PARSER = argparse.ArgumentParser()
PARSER.add_argument("source_path",
                    type=str,
                    help="Source embedding file path")
PARSER.add_argument("target_path",
                    type=str,
                    help="Target embedding file path")
PARSER.add_argument("--output",
                    default="./mapping.h5",
                    type = str,
                    help="Model output path")
ARGS = PARSER.parse_args()

def train_mapping(source_data, target_data):
    print(source_data.shape, target_data.shape)
    model = Sequential()
    model.add(Dense(units=target_data.shape[1], input_shape=(source_data.shape[1], ), use_bias=True))
    model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
              metrics=['mse', 'mae'])
    model.fit(source_data, target_data, epochs=400, batch_size=64)
    model.save(ARGS.output)
    return model

def load_embedding(fp):
    id_to_emb = {}
    with open(fp, 'rt') as fin:
        fin.readline()
        for line in fin:
            id_, *emb = line.strip().split()
            emb = [float(element) for element in emb]
            id_to_emb[id_] = emb
    return id_to_emb

if __name__ == "__main__":
    id_to_hpe_emb = load_embedding(ARGS.source_path)
    id_to_line_emb = load_embedding(ARGS.target_path)
    hpe_embeddings = []
    line_embeddings = []
    for id_ in set(id_to_hpe_emb.keys()).intersection(set(id_to_line_emb.keys())):
        hpe_embeddings.append(id_to_hpe_emb[id_])
        line_embeddings.append(id_to_line_emb[id_])
    hpe_embeddings = np.array(hpe_embeddings)
    line_embeddings = np.array(line_embeddings)
    train_mapping(line_embeddings, hpe_embeddings)
