#/bin/bash
RAW_DATA_DIR="/path/to/raw/data"
IM_DIR="./intermediate_data"
TARGET_SOURCE="./target_source"

mkdir $IM_DIR $TARGET_SOURCE
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

cat $RAW_DATA_DIR/entertainment_events_20180523.csv $RAW_DATA_DIR/entertainment_events_20180903.csv $RAW_DATA_DIR/entertainment_events_v6.csv $RAW_DATA_DIR/entertainment_events_v7.csv | sort -u > $IM_DIR/entertainment_events_all.csv
python3 ./preprocessing/parser.py --testing 1 $IM_DIR/entertainment_events_all.csv > $TARGET_SOURCE/unseen_2018_events_description.csv

python3 ./preprocessing/parser.py  $RAW_DATA_DIR/entertainment_events_v7.csv > $IM_DIR/events_description_v7.data
awk '{print $2}' $TARGET_SOURCE/user-item.data | sort -u | awk '{print "^"$0","}' > $IM_DIR/training_event_pattern.txt
grep -f $IM_DIR/training_event_pattern.txt $IM_DIR/events_description_v7.data> $TARGET_SOURCE/events.csv
