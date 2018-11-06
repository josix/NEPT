# /bin/sh
USAGE="usage: run.sh training_file_path userId_column_number itemId_column_number title_column_number unseen_events_file_path"
test "$#" != "5" && echo "$USAGE" && exit 1

if ! [ -e $1 ]
then
  echo "$USAGE\nFile \"$1\" is not exists"
  exit 1
fi

if ! [ -f $1 ]
then
  echo "$USAGE\n\"$1\" is not a file"
  exit 1
fi

# Generate user-item graph
./script/extract.sh $1 $2 $3 $4
python3 ./script/generate_item_list.py -o ./data/itemsList.data ./data/user-item.data

# random separate data
#python3 ./script/data_split.py -o1 ./data/train.data -o2 ./data/test.data ./data/itemsList.data

# generate network data for training hpe
python3 "./script/export.py" -o "./data/export.data" ./data/itemsList.data

# HPE trainning
../proNet-core/cli/hpe -train ./data/export.data -save ./data/rep.hpe -undirected 1 -dimensions 128 -reg 0.01 -sample_times 5 -walk_steps 5 -negative_samples 5 -alpha 0.025 -threads 4
# Turn word2vec format into JSON
python3 ./script/rep_transform.py -o ./data/rep.json ./data/rep.hpe
# Segement title
python3 ./script/segement.py -o ./data/tags.json ./data/eventsTitle.data
mkdir data/textrank
# Generate keywords form title and description
python3 ./script/textrank.py -o ./data/textrank/textrank  ./data/eventsTitle.data
# Construct user-label(word) graph
python3 ./script/construct_user_word_graph.py -o ./data/user-label.data ./data/export.data ./data/textrank/textrank.json ./data/textrank/textrank_mapping.txt
# Train line-2nd on user-word graph
../proNet-core/cli/line -train ./data/user-label.data  -save ./data/textrank/rep.line2 -undirected 1 -order 2 -dimensions 128 -sample_times 4 -negative_samples 5 -alpha 0.025 -threads 4
# Propagate HPE embedding based on user-word graph
python3 ./src/label_propagation.py $5 ./data/rep.json ./data/textrank/textrank.json ./data/textrank
