# /bin/sh
USAGE="usage: run_label_embedding.sh source target_dir\n\nsource:\tThe directory of the training data\ntarget_dir:\tThe directory to store trained data"
test "$#" != "2" && echo "$USAGE" && exit 1

if ! [ -e $1 ]
then
  echo "$USAGE\nDirectroy \"$1\" is not exists"
  exit 1
fi
SOURCE_DIR=$1
TARGET_DIR=$2

mkdir $TARGET_DIR
# HPE trainning
./proNet-core/cli/hpe -train $SOURCE_DIR/user-item.data -save $TARGET_DIR/rep.hpe -undirected 1 -dimensions 128 -reg 0.01 -sample_times 5 -walk_steps 5 -negative_samples 5 -alpha 0.025 -threads 20
# Turn word2vec format into JSON
python3 ./script/rep_transform.py -o $TARGET_DIR/rep.json $TARGET_DIR/rep.hpe
mkdir $TARGET_DIR/textrank
# Generate keywords form title and description
python3 ./script/textrank.py -o $TARGET_DIR/textrank/textrank  $SOURCE_DIR/events.csv
# Construct user-label(word) graph
python3 ./script/construct_user_word_graph.py -o $TARGET_DIR/textrank/user-label.data $SOURCE_DIR/user-item.data $TARGET_DIR/textrank/textrank.json $TARGET_DIR/textrank/textrank_mapping.txt
# Train line-2nd on user-word graph
./proNet-core/cli/line -train $TARGET_DIR/textrank/user-label.data  -save $TARGET_DIR/textrank/rep.line2 -undirected 1 -order 2 -dimensions 128 -sample_times 40 -negative_samples 5 -alpha 0.025 -threads 20
# Generate semantic space embedding
python3 ./src/label_propagation.py $SOURCE_DIR/unseen_2018_events_description.csv $TARGET_DIR/rep.json $TARGET_DIR/textrank/textrank.json $TARGET_DIR/textrank
# Generate preference space embedding
python3 ./src/vsm_propagation.py $SOURCE_DIR/unseen_2018_events_description.csv $TARGET_DIR/rep.json  $TARGET_DIR/textrank/textrank.json
