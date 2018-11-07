from collections import defaultdict
from math import sqrt
import subprocess
import random

def load_events(path):
    with open(path, 'rt') as fin:
        training_id = []
        for line in fin:
            user, item, title = line.strip().split(',')
            training_id.append(item)
        item_set =  set(training_id)
    return item_set

def load_embedding(path):
    user_vertex_embedding = defaultdict(list)
    item_vertex_embedding = defaultdict(list)
    with open(path, 'rt') as fin:
        fin.readline()
        for line in fin:
            vertex, *embedding = line.strip().split()
            if vertex[0] == 'u':
                user_vertex_embedding[vertex] = [float(x) for x in embedding]
            else:
                item_vertex_embedding[vertex] = [float(x) for x in embedding]
    return user_vertex_embedding, item_vertex_embedding

def random_embedding(unseen_events_set, dim):
    unseen_vertex_embedding_dict = defaultdict(list)
    for item in unseen_events_set:
        unseen_vertex_embedding_dict[item] = [random.uniform(-1, 1) for _ in range(dim)]
    return unseen_vertex_embedding_dict

def recommendation(query, item_vertex_embedding, item_detail_map=None):
    print("query event:", query)
    print(item_detail_map[query])
    if query not in item_vertex_embedding: # Because the number of items for each user is too low so that not all items are embedding
        return False
    query_embedding = item_vertex_embedding[query]
    recommendation_list = []
    for item in item_vertex_embedding:
        cosine_similarity = cosine(query_embedding, item_vertex_embedding[item])
        recommendation_list.append((cosine_similarity, item))
    recommendation_list.sort(reverse=True)
    for index, recommendation in enumerate(recommendation_list[1:11]):
        print("{} Recommendation: {}".format(index, recommendation))
        # show detail
        print(item_detail_map[recommendation[1]])
    return list(map(lambda x: x[1], recommendation_list[1:11]))

def cosine(v1, v2):
    numerator = 0
    denominator = 0
    x_square_summation = 0
    y_square_summation = 0
    for x, y in zip(v1, v2):
        numerator += x*y
        x_square_summation += x**2
        y_square_summation += y**2
    denominator = sqrt(x_square_summation) * sqrt(y_square_summation)
    return numerator / denominator

def eval_unseen_events_num(rec_list, seen_set, unseen_set):
    seen_count = 0
    unseen_count = 0
    for item in rec_list:
        if item in seen_set:
            seen_count += 1
        if item in unseen_set:
            unseen_count += 1
    return (seen_count, unseen_count)

def random_recommendation(query, item_set, item_detail_map=None):
    print('query event:', query)
    print(item_detail_map[query])
    recommendation_list = random.sample(list(item_set), 10)
    for index, recommendation in enumerate(recommendation_list):
        print("{} Recommendation: {}".format(index, recommendation))
        # show detail
        print(item_detail_map[recommendation])
    return recommendation_list

if __name__ == "__main__":
    # show detail
    command = "awk -F, '{print $0 }' '../../kktix/preproecessed_data/eventDetailMap_v7.csv'"
    result = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    item_detail_map = {i.split(',')[0]:i for i in result}

    seen_events = load_events('../source/entertainment_transactions_v7_Before20161231.data')
    unseen_events = load_events('../source/entertainment_transactions_v7_After20161231.data')

    # Embedding Generation
    # model case
    user_vertex_embedding, item_vertex_embedding = load_embedding('../data/rep.hpe')
    _, unseen_vectex_embedding = load_embedding('../unssen_events_rep_hpe2.txt')

    # rendom giving case
    # unseen_vectex_embedding = random_embedding(unseen_events, 128)

    item_vertex_embedding = {**item_vertex_embedding, **unseen_vectex_embedding}

    with open('./data/random_event_query.txt', 'rt') as fin:
        count = 0
        acu_score = 0.0
        acu_recommendation_list = []
        for line in fin:
            count += 1
            # early stop
            if count == 500:
                break
            query = line.strip()

            # Recommendation Model
            # model version
            recommendation_list = recommendation(query, item_vertex_embedding, item_detail_map)

            # random_recommendation
            # recommendation_list = random_recommendation(query, seen_events | unseen_events, item_detail_map)

            if not recommendation_list:
                count -= 1
                continue
            acu_recommendation_list.extend(recommendation_list)

            seen_num, unseen_num = eval_unseen_events_num(recommendation_list, seen_events, unseen_events)
            print('seen_num: {}'.format(seen_num), 'unseen_num: {}'.format(unseen_num), 'unseen_num/total: {}'.format(unseen_num/len(recommendation_list)))
            acu_score += unseen_num/len(recommendation_list)
            print()
        print('ave_score: {}'.format(acu_score/count))
        print('|unique_recommendation| / |all_events|: {}'.format(len(set(acu_recommendation_list))/len(seen_events | unseen_events)))
