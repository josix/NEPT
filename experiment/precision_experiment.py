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

def load_watch_list(path):
    user_watch_list = defaultdict(list)
    with open(path, 'rt') as fin:
        for line in fin:
            user, *items = line.strip().split()
            user_watch_list["u"+user] += items
    return user_watch_list

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

def random_recommendation(query, item_set, item_detail_map=None):
    print('query event:', query)
    print(item_detail_map[query])
    recommendation_list = random.sample(list(item_set), 10)
    for index, recommendation in enumerate(recommendation_list):
        print("{} Recommendation: {}".format(index, recommendation))
        # show detail
        print(item_detail_map[recommendation])
    return recommendation_list


def popularity_recommendation(query, recommendation_list, item_detail_map=None):
    print('query event:', query)
    print(item_detail_map[query])
    for index, recommendation in enumerate(recommendation_list):
        print("{} Recommendation: {}".format(index, recommendation))
        print(item_detail_map[recommendation])
    return recommendation_list


if __name__ == "__main__":
    # show detail
    command = "awk -F, '{print $0 }' '../../kktix/preproecessed_data/eventDetailMap_v7.csv'"
    result = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    item_detail_map = {i.split(',')[0]:i for i in result}

    user_watch_list = load_watch_list('./data/precision/user_unseen_answer.data')

    # random recommendation
    # seen_events = load_events('../source/entertainment_transactions_v7_Before20161231.data')
    # unseen_events = load_events('../source/entertainment_transactions_v7_After20161231.data')

    # model_recommendation
    # hpe/mf + vsm
    # user_vertex_embedding, item_vertex_embedding = load_embedding('../data/rep.hpe')
    # _, unseen_vectex_embedding = load_embedding('../unssen_events_rep.txt')
    # rec_embedding = {**item_vertex_embedding, **unseen_vectex_embedding}

    # GraphSAGE
    _, rec_embedding = load_embedding('../graphSAGE_data/graphsage_mean_small_0.000010/rep.graphsage')

    # popularity_recommendation
    # command = "cat ../source/entertainment_transactions_v7_Before20161231.data ../source/entertainment_transactions_v7_After20161231.data\
    #         | awk -F, 'BEGIN{item[$2]=0}{item[$2] = item[$2] + 1}END{for(i in item){print i, item[i]}}'"
    # result = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    # popularity_list = [(int(i.split()[1]), i.split()[0]) for i in result if len(i.split()) == 2]
    # popularity_list.sort(reverse=True)
    # popularity_list = list(map(lambda x: x[1], popularity_list[:10]))

    # Read experiment data
    with open('./data/user_item_query.txt') as fin:
        count = 0
        maching_count = 0
        total_avep = 0
        for line in fin:
            count += 1
            # early stop
            if count == 500:
                break
            user, item = line.strip().split()

            if len(user_watch_list[user]) == 0:
                count -= 1
                continue

            print('query user:', user)
            # model_recommendation
            recommendation_list = recommendation(item, rec_embedding, item_detail_map)

            # random_recommendation
            # recommendation_list = random_recommendation(item, seen_events | unseen_events, item_detail_map)

            # popularity_recommendation
            # recommendation_list = popularity_recommendation(item, popularity_list, item_detail_map)

            if not recommendation_list:
                # print("No existed query embedding in training data.")
                count -= 1
                continue

            # scoring
            machingNum = len(set(recommendation_list) & set(user_watch_list[user]))
            precision = machingNum/10
            recall = machingNum/len(user_watch_list[user])
            print('precision@10: {} recall@10: {}'.format(precision, recall))
            if recall and precision:
                fscore = 2 / (1 / precision + 1/recall)
                print('F1: {}'.format(fscore))

                ranked_precision = 0
                ranked_machingNum = 0
                for index, item in enumerate(recommendation_list):
                    if item in user_watch_list[user]:
                        ranked_machingNum += 1
                        ranked_precision += ranked_machingNum / (index + 1)
                ave_precision = ranked_precision / ranked_machingNum
                total_avep += ave_precision
                maching_count += 1
                print('AvePrecision: {}\n'.format(ave_precision))
            else:
                print()

        print('# of queries: {}'.format(count))
        print('# of queries(precision > 0): {}'.format(maching_count))
        if maching_count:
            print('MAP: {}'.format(total_avep / maching_count))
