from collections import defaultdict
from math import sqrt
from itertools import combinations
import subprocess
import pickle
import random
import concurrent.futures as cf
import sys
import argparse

from fuzzywuzzy import fuzz
from annoy import AnnoyIndex

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--random",
                    type=bool,
                    default=False,
                    help="Using random recommendation.")
PARSER.add_argument("--embedding",
                    type=bool,
                    default=False,
                    help="Using embedding recommendation.")
PARSER.add_argument("--annoy",
                    type=str,
                    help="Using the file to build annoy and find nearest-neighbor recommendation.")
PARSER.add_argument("--single",
                    type=str,
                    help="Using single embedding recommendation.")
PARSER.add_argument("--concat",
                    nargs='+',
                    type=str,
                    help="Using concat embedding recommendation.")
ARGS = PARSER.parse_args()
def load_events(path):
    with open(path, 'rt') as fin:
        training_id = []
        for line in fin:
            item, *_ = line.strip().split(',')
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
            # print("ERROR ZeroDivisionError: ")
            # print(query, item)
            # print(query_embedding, item_vertex_embedding[item][0])
            cosine_similarity = -100
            #sys.exit()
        recommendation_list.append((cosine_similarity, item))
    # only recommendate new envent, if not comment this line
    recommendation_list = [recommendation for recommendation in recommendation_list if item_vertex_embedding[recommendation[1]][1] != 'hpe']
    recommendation_list.sort(reverse=True)
    return (query, list(map(lambda x: x[1], recommendation_list)))

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

def random_recommendation(query, item_set):
    recommendation_list = random.sample(list(item_set), 10)
    return (query, recommendation_list)

def annoy_recommend(query):
    global annoy_index
    ranking_list = annoy_index.get_nns_by_item(int(query), 10, search_k=-1, include_distances=True)
    return (query, [str(item) for item in ranking_list[0]])

def popularity_recommendation(query, recommendation_list, item_detail_map=None):
    print('query event:', query)
    print(item_detail_map[query])
    for index, recommendation in enumerate(recommendation_list):
        print(f"{index} Recommendation: {recommendation}")
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
    # Past result
    # PAST_RESULT = pickle.load(open('./result/precision@5_2018_transaction_top300_popular_queries_only_new/span3_iter300/eyeball/past_query_log.pkl', 'rb'))
    # PAST_RESULT_LABEL = pickle.load(open('./result/precision@5_2018_transaction_top300_popular_queries_only_new/span3_iter300/eyeball/past_query_log_label.pkl', 'rb'))

    # model_recommendation
    # hpe/mf + vsm
    unseen_vectex_embedding_rank = None
    if ARGS.single:
        # user_vertex_embedding, item_vertex_embedding = load_embedding('../log_transaction_data/rep.hpe')
        _, unseen_vectex_embedding = load_embedding(ARGS.single)
    elif ARGS.concat:
        _, unseen_vectex_embedding_rank = load_embedding(ARGS.concat[0])
        _, unseen_vectex_embedding_tfidf = load_embedding(ARGS.concat[1])
        unseen_vectex_embedding = \
            {key : unseen_vectex_embedding_rank[key] + unseen_vectex_embedding_tfidf[key]
                for key in unseen_vectex_embedding_rank.keys()}
    elif ARGS.annoy:
        _, unseen_vectex_embedding = load_embedding(ARGS.annoy)
        annoy_index = AnnoyIndex(len(list(unseen_vectex_embedding.values())[0]))
        for id_, vector in unseen_vectex_embedding.items():
            annoy_index.add_item(int(id_), vector)
        annoy_index.build(10)

    #rec_embedding = {**{ key:(value, 'hpe') for key, value in item_vertex_embedding.items() },
    #                 **{ key:(value, 'propagation') for key, value in unseen_vectex_embedding.items()} }
    rec_embedding = { key:(value, 'propagation') for key, value in unseen_vectex_embedding.items()}

    # model_recommendation
    with cf.ProcessPoolExecutor(max_workers=20) as executor:
        future_to_user = None
        if ARGS.random:
            # random recommendation
            unseen_events = load_events('../source/unseen_2018_events_description.csv')
            future_to_user =\
                    {executor.submit(random_recommendation, query, unseen_events) : user
                    for index, (user, query) in enumerate(query_gen(user_watch_list, rec_embedding, './data/precision@5_1user_1item_top100_popular_query_user_click_10.txt'))
                    #for index, (user, query) in enumerate(query_gen(user_watch_list, rec_embedding, './data/precision@5_1user_1item_top300_popular_query_2018.txt'))
                    if index <= 499}
        elif ARGS.embedding:
            future_to_user =\
                    {executor.submit(recommend, query, rec_embedding) : user
                    for index, (user, query) in enumerate(query_gen(user_watch_list, rec_embedding, './data/precision@5_1user_1item_top100_popular_query_user_click_10.txt'))
                    #for index, (user, query) in enumerate(query_gen(user_watch_list, rec_embedding, './data/precision@5_1user_1item_top300_popular_query_2018.txt'))
                    if index <= 499}
        elif ARGS.annoy:
            future_to_user =\
                    {executor.submit(annoy_recommend, query) : user
                    for index, (user, query) in enumerate(query_gen(user_watch_list, rec_embedding, './data/precision@5_1user_1item_top100_popular_query_user_click_10.txt'))
                    #for index, (user, query) in enumerate(query_gen(user_watch_list, rec_embedding, './data/precision@5_1user_1item_top300_popular_query_2018.txt'))
                    if index <= 499}

        count = 0
        maching_count = 0
        total_avep = 0
        total_ave_distance_query_to_rec = 0
        total_ave_distance_rec_to_rec = 0
        total_rec_num = 0
        seen_query = set()
        cases_to_result = {}
        all_recommendation_set = set()
        # Past result
        past_avep = 0
        past_avep_old = 0
        label_count = 0
        for future in cf.as_completed(future_to_user):
            count += 1
            user = future_to_user[future]
            query_item, recommendation_list = future.result()
            top_5_recommendation_list = recommendation_list[1:6]
            print('user:', user)
            print('query event:', query_item)
            print(item_detail_map[query_item])
            for index, recommendation in enumerate(top_5_recommendation_list):
                print(f"{index} Recommendation: {recommendation}")
                # show detail
                print(item_detail_map[recommendation])

            # Past result
            # print('='*20)
            # rerank_past_result = []
            # for item in PAST_RESULT[query_item]:
            #     rerank_past_result.append((recommendation_list.index(item) - 1, item))
            # rerank_past_result.sort()
            # for _, item in rerank_past_result:
            #     print(item_detail_map[item])
            # print()
            # print("For old items:")
            # print("{:50s}{:50s}".format("Current result", "Past result"))
            # ap = 0
            # score = 0
            # ap_old = 0
            # score_old = 0
            # for index, ((rerank, item), (item_old)) in enumerate(zip(rerank_past_result, PAST_RESULT[query_item])):
            #     print("{0:50s}{1:50s}".format(str(rerank)+" Recommendation: "+ str(item), str(index)+" Recommendation: "+ str(item_old)))
            #     # print("(original_rank: {}, label: {})".format(rerank , item, original_rank, PAST_RESULT_LABEL[(query_item, item)]))
            #     print("{0:50s}{1:50s}".format(PAST_RESULT_LABEL[(query_item, item)], PAST_RESULT_LABEL[(query_item, item_old)]))
            #     if PAST_RESULT_LABEL[(query_item, item)] != "" and PAST_RESULT_LABEL[(query_item, item_old)] != "":
            #         score += 1 if 'O' in PAST_RESULT_LABEL[(query_item, item)] else 0
            #         ap += (score  / (index+1)) / 5
            #         score_old += 1 if 'O' in PAST_RESULT_LABEL[(query_item, item_old)] else 0
            #         ap_old += (score_old / (index+1)) / 5
            # if ap > 0 and ap_old > 0:
            #     label_count += 1
            # if 1 > ap > 0 and 1 > ap_old > 0:
            #     print("Both O and X")
            # print("Average Precision:")
            # print("{0:<50f}{1:<50f}".format(ap, ap_old))
            # past_avep += ap
            # past_avep_old += ap_old

            # Coverage Scoring
            all_recommendation_set = all_recommendation_set | set(top_5_recommendation_list)
            if query_item not in seen_query:
                seen_query.add(query_item)
                total_rec_num += len(top_5_recommendation_list)

            # Edit Distance Scoring
            edit_distance_query_to_rec = \
                    [fuzz.ratio(item_detail_map[query_item], item_detail_map[rec_item]) for rec_item in top_5_recommendation_list]
            ave_edit_distance_query_to_rec = sum(edit_distance_query_to_rec) / len(edit_distance_query_to_rec)
            total_ave_distance_query_to_rec += ave_edit_distance_query_to_rec
            print('='*20)
            print("Levenshtein Distance:")
            print(f"Average Distance (query to rec): {ave_edit_distance_query_to_rec}")
            edit_distance_rec_to_rec = \
                    [fuzz.ratio(item_detail_map[rec_item_first], item_detail_map[rec_item_second])\
                     for rec_item_first, rec_item_second in combinations(top_5_recommendation_list, 2)]
            ave_edit_distance_rec_to_rec = sum(edit_distance_rec_to_rec) / len(edit_distance_rec_to_rec)
            total_ave_distance_rec_to_rec += ave_edit_distance_rec_to_rec
            print(f"Average Distance (rec to rec): {ave_edit_distance_rec_to_rec}")

            # Precision and Recall Scoring
            print('='*20)
            machingNum = len(set(top_5_recommendation_list) & set(user_watch_list[user]))
            if machingNum:
                print(
                    f'hit eventIDs: {set(top_5_recommendation_list) & set(user_watch_list[user])}'
                )

                print('='*20)
            precision = machingNum/5
            recall = machingNum/len(user_watch_list[user])
            print(f'precision@5: {precision} recall@5: {recall}')
            cases_to_result[(user, query_item)] = {
                    'precision': precision,
                    'recall': recall,
                    'iter_result_distance': ave_edit_distance_rec_to_rec,
                    'result_query_distance': ave_edit_distance_query_to_rec,
                    }
            if recall and precision:
                fscore = 2 / (1 / precision + 1/recall)
                print(f'F1: {fscore}')

                ranked_precision = 0
                ranked_machingNum = 0
                for index, rec_item in enumerate(top_5_recommendation_list):
                    if rec_item in user_watch_list[user]:
                        ranked_machingNum += 1
                        ranked_precision += ranked_machingNum / (index + 1)
                ave_precision = ranked_precision / ranked_machingNum
                total_avep += ave_precision
                maching_count += 1
                print(f'AvePrecision: {ave_precision}\n')
            else:
                print()

#  FOR COMPARE DIFFERENT METHOD ON CERTAIN METRIC
#    with open('./result/precision@5_2018_transaction_top300_popular_queries_only_new/span3_iter300/textrank.pkl', 'wb') as fout:
#        pickle.dump(cases_to_result, fout)
#
    print(f'# of queries: {count}')
    print(f'# of queries(precision > 0): {maching_count}')
    if count:
        print('Mean Average Edit Distance (query, recommendation): {:.2f}%'.format(total_ave_distance_query_to_rec / count))
        print('Mean Average Edit Distance (recommendation, recommendation): {:.2f}%'.format(total_ave_distance_rec_to_rec / count))
    print(f'Coverage Rate: {len(all_recommendation_set) / total_rec_num}')
    if maching_count:
        print(f'MAP: {total_avep / maching_count}')
        # Past result
        # print('MAP_on_old_rec: {}'.format(past_avep /  label_count))
        # print('MAP_on_old_rec_past: {}'.format(past_avep_old /  label_count))
