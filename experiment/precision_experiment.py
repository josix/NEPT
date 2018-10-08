from collections import defaultdict
from math import sqrt
from itertools import combinations
import subprocess
import random
import concurrent.futures as cf
import sys

from fuzzywuzzy import fuzz

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
            if user[0] == "u":
                user_watch_list[user[1:]] += items
            else:
                user_watch_list[user] += items
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

def recommend(query, item_vertex_embedding):
    query_embedding = item_vertex_embedding[query][0]
    recommendation_list = []
    for item in item_vertex_embedding:
        try:
            cosine_similarity = cosine(query_embedding, item_vertex_embedding[item][0])
        except ZeroDivisionError:
            print("ERROR ZeroDivisionError: ")
            print(query, item)
            print(query_embedding, item_vertex_embedding[item][0])
            sys.exit()
        recommendation_list.append((cosine_similarity, item))
    # only recommendate new envent, if not comment this line
    recommendation_list = [recommendation for recommendation in recommendation_list if item_vertex_embedding[recommendation[1]][1] != 'hpe']
    recommendation_list.sort(reverse=True)
    return (query, list(map(lambda x: x[1], recommendation_list[1:6])))

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

def query_gen(user_watch_list, item_vertex_embedding, fp):
    # Filtering query list
    with open(fp) as fin:
        for line in fin:
            user, query_item = line.strip().split()
            if user_watch_list[user] == [] or query_item not in item_vertex_embedding:
                continue
            yield user, query_item


if __name__ == "__main__":
    # show detail
    command = "awk -F, '{print $0 }' '../../kktix/preproecessed_data/eventDetailMap_v7.csv' '../../kktix/preproecessed_data/eventDetailMap_20180903.csv' '../../kktix/preproecessed_data/eventDetailMap_20180523.csv'"
    result = subprocess.check_output(command, shell=True, encoding='utf-8').split('\n')
    item_detail_map = defaultdict(lambda: "No corresponding metadata in entertainment_events_*.csv")
    for i in result:
        item_detail_map[i.split(',')[0]] = i

    user_watch_list = load_watch_list('./data/precision/transaction_future_answer.data')

    # random recommendation
    # seen_events = load_events('../source/entertainment_transactions_v7_Before20161231.data')
    # unseen_events = load_events('../source/entertainment_transactions_v7_After20161231.data')

    # model_recommendation
    # hpe/mf + vsm
    user_vertex_embedding, item_vertex_embedding = load_embedding('../log_data/rep.hpe')
    _, unseen_vectex_embedding = load_embedding('../log_transaction_data/unseen_data/unseen_events_rep_hpe(textrank_similarity).txt')
    rec_embedding = {**{ key:(value, 'hpe') for key, value in item_vertex_embedding.items() },
                     **{ key:(value, 'propagation') for key, value in unseen_vectex_embedding.items()} }

    # GraphSAGE
    # _, rec_embedding = load_embedding('../graphSAGE_data/graphsage_mean_small_0.00001_256/rep.graphsage')

    # popularity_recommendation
    # command = "cat ../source/entertainment_transactions_v7_Before20161231.data ../source/entertainment_transactions_v7_After20161231.data\
    #         | awk -F, 'BEGIN{item[$2]=0}{item[$2] = item[$2] + 1}END{for(i in item){print i, item[i]}}'"
    # result = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    # popularity_list = [(int(i.split()[1]), i.split()[0]) for i in result if len(i.split()) == 2]
    # popularity_list.sort(reverse=True)
    # popularity_list = list(map(lambda x: x[1], popularity_list[:10]))

    # random_recommendation
    # recommendation_list = random_recommendation(query_item, seen_events | unseen_events, item_detail_map)

    # popularity_recommendation
    # recommendation_list = popularity_recommendation(query_item, popularity_list, item_detail_map)

    # model_recommendation
    with cf.ProcessPoolExecutor(max_workers=4) as executor:
        future_to_user =\
                {executor.submit(recommend, query, rec_embedding) : user
                for index, (user, query) in enumerate(query_gen(user_watch_list, rec_embedding, './data/precision@5_1user_1item_transaction_future_query.txt'))
                if index <= 499}

        count = 0
        maching_count = 0
        total_avep = 0
        total_ave_distance_query_to_rec = 0
        total_ave_distance_rec_to_rec = 0
        total_rec_num = 0
        seen_query = set()
        all_recommendation_set = set()
        for future in cf.as_completed(future_to_user):
            count += 1
            user = future_to_user[future]
            query_item, recommendation_list = future.result()
            print('user:', user)
            print('query event:', query_item)
            print(item_detail_map[query_item])
            for index, recommendation in enumerate(recommendation_list):
                print("{} Recommendation: {}".format(index, recommendation))
                # show detail
                print(item_detail_map[recommendation])

            # Coverage Scoring
            all_recommendation_set = all_recommendation_set | set(recommendation_list)
            if query_item not in seen_query:
                seen_query.add(query_item)
                total_rec_num += len(recommendation_list)

            # Edit Distance Scoring
            edit_distance_query_to_rec = \
                    [fuzz.ratio(item_detail_map[query_item], item_detail_map[rec_item]) for rec_item in recommendation_list]
            ave_edit_distance_query_to_rec = sum(edit_distance_query_to_rec) / len(edit_distance_query_to_rec)
            total_ave_distance_query_to_rec += ave_edit_distance_query_to_rec
            print('='*20)
            print("Levenshtein Distance:")
            print("Average Distance (query to rec): {}".format(ave_edit_distance_query_to_rec))
            edit_distance_rec_to_rec = \
                    [fuzz.ratio(item_detail_map[rec_item_first], item_detail_map[rec_item_second])\
                     for rec_item_first, rec_item_second in combinations(recommendation_list, 2)]
            ave_edit_distance_rec_to_rec = sum(edit_distance_rec_to_rec) / len(edit_distance_rec_to_rec)
            total_ave_distance_rec_to_rec += ave_edit_distance_rec_to_rec
            print("Average Distance (rec to rec): {}".format(ave_edit_distance_rec_to_rec))

            # Precision and Recall Scoring
            print('='*20)
            machingNum = len(set(recommendation_list) & set(user_watch_list[user]))
            if machingNum:
                print('hit eventIDs: {}'.format(set(recommendation_list) & set(user_watch_list[user])))
                print('='*20)
            precision = machingNum/5
            recall = machingNum/len(user_watch_list[user])
            print('precision@5: {} recall@5: {}'.format(precision, recall))
            if recall and precision:
                fscore = 2 / (1 / precision + 1/recall)
                print('F1: {}'.format(fscore))

                ranked_precision = 0
                ranked_machingNum = 0
                for index, rec_item in enumerate(recommendation_list):
                    if rec_item in user_watch_list[user]:
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
    print('Mean Average Edit Distance (query, recommendation): {:.2f}%'.format(total_ave_distance_query_to_rec / count))
    print('Mean Average Edit Distance (recommendation, recommendation): {:.2f}%'.format(total_ave_distance_rec_to_rec / count))
    print('Coverage Rate: {}'.format(len(all_recommendation_set) / total_rec_num))
    if maching_count:
        print('MAP: {}'.format(total_avep / maching_count))
