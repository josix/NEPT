#! /bin/sh
USAGE="usage: experiment.sh result_dir\nresult_dir:\tThe directory to store experiment result."
test "$#" != "1" && echo "$USAGE" && exit 1

EXPERIMENT_RESULT=$1


mkdir $EXPERIMENT_RESULT
# Experiment
python3 precision_experiment.py  --annoy ../tfidf_vsm.txt > $EXPERIMENT_RESULT/tfidf_vsm.txt
python3 precision_experiment.py --embedding 1 --concat ../unseen_events_label_embedding\(textrank_top100queries_strong_user_before2018\).txt ../unssen_events_rep_hpe\(tfidf_2018unseen_top100queries_strong_user_before2018\).txt > $EXPERIMENT_RESULT/concat.txt
python3 precision_experiment.py --random 1  --concat ../unseen_events_label_embedding\(textrank_top100queries_strong_user_before2018\).txt ../unssen_events_rep_hpe\(tfidf_2018unseen_top100queries_strong_user_before2018\).txt  > $EXPERIMENT_RESULT/random.txt
python3 precision_experiment.py --embedding 1 --single ../mf.txt > $EXPERIMENT_RESULT/mf.txt
python3 precision_experiment.py --embedding 1 --single ../hpe.txt > $EXPERIMENT_RESULT/hpe.txt
