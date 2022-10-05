import random
from math import exp

def load_embedding(fp):
    id_to_emb = {}
    with open(fp, 'rt') as fin:
        fin.readline()
        for line in fin:
            id_, *emb = line.strip().split()
            emb = [[float(element) for element in emb]]
            id_to_emb[id_] = emb
    return id_to_emb

def sigmoid(s):
    return 1 / (1 + exp(-s))

def matrix_multiply(m1, m2):
    a_row, a_col = len(m1), len(m1[0])
    b_row, b_col = len(m2), len(m2[0])
    result = []
    for row in range(a_row):
        new_row = [
            sum(m1[row][i] * m2[i][col] for i in range(a_col))
            for col in range(b_col)
        ]

        result.append(new_row)
    return result

def dot_product(m1, m2):
    a_row, a_col = len(m1), len(m1[0])
    b_row, b_col = len(m2), len(m2[0])
    if a_row != b_row:
        print('wrong dim')
        return None
    return [[sum(a*b for a, b in zip(m1[row], m2[row]))] for row in range(a_row)]

def matrix_add(m1, m2):
    a_row, a_col = len(m1), len(m1[0])
    b_row, b_col = len(m2), len(m2[0])
    if a_row != b_row:
        print('wrong dim')
        return None
    result = []
    for row in range(a_row):
        new_row = [a+b for a, b in zip(m1[row], m2[row])]
        result.append(new_row)
    return result

def rmse(m1, m2):
    a_row, a_col = len(m1), len(m1[0])
    b_row, b_col = len(m2), len(m2[0])
    if a_row != b_row:
        print('wrong dim')
        return None
    result = []
    for row in range(a_row):
        new_row = (sum((a-b)**2 for a, b in zip(m1[row], m2[row])) / a_col)**0.5
        result.append([new_row])
    return result

def train_mapping(source_space, target_space, dim=(128, 128), lr=0.01, max_iters=10000):
    intersect_ids:set = set(source_space.keys()).intersection(target_space.keys())
    weights =  [[random.uniform(-0.5, 0.5) for _ in range(dim[1])] for _ in range(dim[0])]
    precision = 0.0001
    previous_step_size = 1
    iters = 0
    while iters < max_iters:
        loss = 0
        for id_ in intersect_ids:
            source_emb = source_space[id_]
            target_emb = target_space[id_]
            print(source_emb, target_emb, weights)
            # forward
            predict_emb = matrix_multiply(source_emb, weights)
            print(predict_emb)
            # compute loss
            loss += rmse(target_emb, predict_emb)[0][0]
            # update
            if previous_step_size > precision:
                update_gradient = [[-1 * lr * 0.5 * (1/loss) * x  for x in source_emb[0]] for _ in range(len(weights))]
                prev_weight = weights
                weights = matrix_add(weights, update_gradient)
                # previous_step_size = distance(weight, prev_weight)
        print(f'loss {loss / len(intersect_ids)}')
        iters += 1
            #break
    return weights

if __name__ == "__main__":
    id_to_hpe_emb = load_embedding('../log_transaction_data/rep.hpe')
    id_to_line_emb = load_embedding('../log_transaction_data/textrank_vsm/rep.line2')
    print(len(id_to_hpe_emb), len(id_to_line_emb))
    id_to_hpe_emb = {1: [[0, 0]], 2: [[1,1]], 3: [[2, 2]], 4:[[3,3]], 5:[[4, 4]], 6:[[5, 5]]}
    id_to_line_emb={6: [[6, 6]], 1: [[1,1]], 2: [[2, 2]], 3:[[3,3]], 4:[[4, 4]], 5:[[5, 5]]}
    model = train_mapping(id_to_line_emb, id_to_hpe_emb, dim=(2,2))
