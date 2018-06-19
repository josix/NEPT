from collections import defaultdict
from math import sqrt
import subprocess
import random

def load_raw_data(training_path, testing_path):
    user_set = set()
    user_watch_list = defaultdict(list)
    item_set = set()
    with open(training_path, 'rt') as fin:
        for line in fin:
            user, *items = line.strip().split()
            user_set = user_set | {"u"+user}
            item_set = item_set | set(items)
            # user_watch_list["u"+user] += items # Only use testing data for experiment answer simulation
    with open(testing_path, 'rt') as fin:
        for line in fin:
            user, *items = line.strip().split()
            user_set = user_set | {"u"+user}
            item_set = item_set | set(items)
            user_watch_list["u"+user] += items
    return user_watch_list, user_set, item_set

def load_embedding(path, user_set, item_set):
    user_vertex_embedding = defaultdict(list)
    item_vertex_embedding = defaultdict(list)
    with open(path, 'rt') as fin:
        for line in fin:
            vertex, *embedding = line.strip().split()
            if vertex in user_set:
                user_vertex_embedding[vertex] = [float(x) for x in embedding]
            elif vertex in item_set:
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
    command = "awk -F, '{print $0 }' '../kktix/preproecessed_data/eventDetailMap_v7.csv'"
    result = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    item_detail_map = {i.split(',')[0]:i for i in result}

    user_watch_list, user_set, item_set = load_raw_data('../kktix/netWorkDataAllState/training_data.data', "../kktix/netWorkDataAllState/testing_data.data")
    # model_recommendation
    # user_vertex_embedding, item_vertex_embedding = load_embedding('../kktix/netWorkDataAllState/rep.hpe', user_set, item_set)

    # popularity_recommendation
    command = "cat ../kktix/netWorkDataAllState/etNet.test ../kktix/netWorkDataAllState/etNet.train\
            | awk 'BEGIN{item[$2]=0}{item[$2] = item[$2] + 1}END{for(i in item){print i, item[i]}}'"
    result = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    popularity_list = [(int(i.split()[1]), i.split()[0]) for i in result if len(i.split()) == 2]
    popularity_list.sort(reverse=True)
    popularity_list = list(map(lambda x: x[1], popularity_list[:10]))

    # Read experiment data
    with open('../kktix/netWorkDataAllState/experimentQuery.data') as fin:
        count = 0
        maching_count = 0
        total_avep = 0
        for line in fin:
            count += 1
            # early stop
            if count == 10:
                break
            user, item = line.strip().split()
            print('query user:', user)
            # model_recommendation
            # recommendation_list = recommendation(item, item_vertex_embedding, item_detail_map)

            # random_recommendation
            # recommendation_list = random_recommendation(item, item_set, item_detail_map)

            # popularity_recommendation
            recommendation_list = popularity_recommendation(item, popularity_list, item_detail_map)

            if not recommendation_list:
                print("No existed query embedding in training data.")
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
