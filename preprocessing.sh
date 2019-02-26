#/bin/bash
RAW_DATA_DIR="/path/to/raw/data"
IM_DIR="./intermediate_data"
TARGET_SOURCE="./target_source"

mkdir $IM_DIR $TARGET_SOURCE
# Generate training user-item data
awk -F, ' BEGIN{
            OFS=","
          }
          {
            if( $11 != "" && $5 != "" && $3 != "" && (index($15, "2015") || index($15, "2016"))){
              print $5, $11, $13 #userID, eventId, eventTitle
            }
        }'  $RAW_DATA_DIR/entertainment_transactions_v7.csv > $IM_DIR/entertainment_transactions_v7_Before20161231.data
awk -f ./preprocessing/extract_user_item_frequency.awk $IM_DIR/entertainment_transactions_v7_Before20161231.data > $IM_DIR/user-item-frequency.data
awk -f ./preprocessing/filter_access.awk $IM_DIR/user-item-frequency.data > $IM_DIR/transaction_export.data

awk -f ./preprocessing/extract_ip_item_frequency.awk /tmp2/KKTeam/KKTIX/kktix_cc_elb_log_201706 > $IM_DIR/kktix_log_201706.data
awk -f ./preprocessing/filter_access.awk $IM_DIR/kktix_log_201706.data > $IM_DIR/log_export.data
cat $IM_DIR/*_export.data > $TARGET_SOURCE/user-item.data

# Generate training (seen) events data
python3 ./preprocessing/parser.py  $RAW_DATA_DIR/entertainment_events_v7.csv > $IM_DIR/events_description_v7.data
awk '{print $2}' $TARGET_SOURCE/user-item.data | sort -u | awk '{print "^"$0","}' > $IM_DIR/training_event_pattern.txt
grep -f $IM_DIR/training_event_pattern.txt $IM_DIR/events_description_v7.data> $TARGET_SOURCE/events.csv

# Generate testing query data
mkdir ./experiment/data
mkdir ./experiment/data/precision
awk -f ./preprocessing/user_to_items.awk $IM_DIR/user-item-frequency.data > $IM_DIR/user_to_items_training.data
cut -f 2,3 -d, $IM_DIR/entertainment_transactions_v7_Before20161231.data | sort -t, -k 2 | uniq -c | sort -r -k 1 -n| head -n 500 > $IM_DIR/popular_events_with_count.txt
awk '{$1=""; print $0}' $IM_DIR/popular_events_with_count.txt| head -n 100 > $IM_DIR/popular_events_100.txt
python3 ./experiment/query_generator/precision_query_generator.py $IM_DIR/user_to_items_training.data $IM_DIR/popular_events_100.txt > ./experiment/data/precision@5_1user_1item_top100_popular_query_user_click_10.txt

# Generate testing (unseen) events data
cat $RAW_DATA_DIR/entertainment_events_20180523.csv $RAW_DATA_DIR/entertainment_events_20180903.csv $RAW_DATA_DIR/entertainment_events_v6.csv $RAW_DATA_DIR/entertainment_events_v7.csv | sort -u > $IM_DIR/entertainment_events_all.csv
python3 ./preprocessing/parser.py --testing 1 $IM_DIR/entertainment_events_all.csv > $IM_DIR/unseen_2018_events_description.csv

# Treat the query item as unseen item
cut -f 2 -d ' '  ./experiment/data/precision@5_1user_1item_top100_popular_query_user_click_10.txt | sort -u > $IM_DIR/query_item.data
grep -f $IM_DIR/query_item.data $TARGET_SOURCE/events.csv >  $IM_DIR/query_item_description.data
cat $IM_DIR/unseen_2018_events_description.csv $IM_DIR/query_item_description.data > $TARGET_SOURCE/unseen_2018_events_description.csv

# Generate testing answer data
awk -f ./preprocessing/extract_user_item_frequency_2018.awk $RAW_DATA_DIR/entertainment_transactions_20180903.csv > $IM_DIR/user-item-2018.data
awk -f ./preprocessing/user_to_items.awk $IM_DIR/user-item-2018.data > ./experiment/data/precision/transaction_future_answer.data

# Generate metadata
mkdir -p ../kktix/preproecessed_data
awk -F, '{OFS=","; print $1, $4, $5}' $RAW_DATA_DIR/entertainment_events_20180903.csv | sort -u > ../kktix/preproecessed_data/eventDetailMap_20180903.csv
awk -F, '{OFS=","; print $1, $4, $5}' $RAW_DATA_DIR/entertainment_events_20180523.csv | sort -u > ../kktix/preproecessed_data/eventDetailMap_20180523.csv
awk -F, 'BEGIN{OFS=","} {print $11,$12,$13}' $RAW_DATA_DIR/entertainment_transactions_v7.csv | sort | uniq  > ../kktix/preproecessed_data/eventDetailMap_v7.csv
