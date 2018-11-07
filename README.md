# NEPT

NEPT is a toolbox used for preparing data to conduct some experiments on [proNet-core](https://github.com/cnclabs/proNet-core)'s algorithm.

NEPT includes features below:
- extract data which are written in csv format
- convert user-item record to user-list(of items) record
- split all data into a training set and a testing set
- export [proNet-core's training data format](https://github.com/cnclabs/proNet-core#task)
- propagate new embedding to unseen data by using vsm model

## Setup Environment
Clone the repo with its submodule
```bash
git clone --recursive https://github.com/wilson8507/NEPT.git
```
Or clone them separately
```bash
git clone https://github.com/wilson8507/NEPT.git
git submodule init
git submodule update
```
After cloning this repo and its submodule

You need to [compile the proNet-core](https://github.com/cnclabs/proNet-core#compilation) and install some python third-party libraries.
```bash
pip install pipenv
pipenv install [--skip-lock]
pipenv shell
```

## Add Source Files to This Repo
Create a source/ directory
```bash
mkdir source
```
Move the log file and unseen title file into the source/ directory

File example:
(No need to keep the header of each column)
```csv
# logFiles.csv
# userId,eventId,eventTitle
0,1,測試活動
0,2,測試活動2
1,2,測試活動2
```
```csv
# unseen_events_file.csv
# id,title
0,MOPCON 2018 Call for Recommendation
1,不費力的發聲技巧～小蛙老師的輕鬆說話課
2,社團法人台灣健康整合服務協會-第三屆會員大會
3,解脫道一、二階-測試
```

## Complete All Things
**Propagation by TFIDF:** ([The full execution flow for propagation by TFIDF](https://https://github.com/wilson8507/NEPT/tree/label_embedding#the-full-execution-flow-for-propagation-by-tfidf))
```
./run.sh <logFiles_path> <userId_column_number> <itemId_column_number> <title_column_number> <unseen_events_file_path>
```
e.g.
```
./run.sh source/entertainment_transactions_v7.csv 5 11 13 source/unseen_events.csv
```

**Propagation by label-embedding:** ([The full execution flow for propagation by label-embedding](https://https://github.com/wilson8507/NEPT/tree/label_embedding#the-full-execution-flow-for-propagation-by-label-embedding))
```
./run_label_embedding.sh <logFiles_path> <userId_column_number> <itemId_column_number> <title_column_number> <unseen_events_file_path>
```
e.g.
```
./run_label_embedding.sh source/entertainment_transactions_v7_Before20161231.data 1 2 3 source/unseen_events_description.csv
```

## The Full Execution Flow for Propagation by TFIDF
1. Execute extact.sh will produce a user-item pairs data.
(The default filename of output is "user-item.data")
```bash
./script/extract.sh <csv_flile_path> <user_column_num> <item_column_num>
```
2. Execute generate_item_list.py will convert user-item pairs data into user-list(of items) pairs.
(The default filename of output is "itemsList.data")
```bash
python3 ./script/generate_item_list.py [-o OUTPUT_PATH] <user-item_file_path>
```
3. Execute data_split.py will produce a training set and a testing set, both of them come from the output file of the previous step. 
(The default filename of outputs are "training.data" and "testing.data")
```bash
python3 ./script/data_split.py [-o1 TRAIN_OUTPUT] [-o2 TEST_OUTPUT] <items-list_file_path>
```
4. Execute export.py will convert user-list(of items) data into the output data in [proNet-core's format](https://github.com/cnclabs/proNet-core#task).
(The default filename of output is "export.data")
```bash
python3 ./script/export.py [-o OUTPUT_PATH] <user-itemslist_file_path>
```
5. Using proNet-core's model to get the embeddings of the users and items.
```bash
./proNet-core/cli/hpe -train ./data/export.data
                      -save ./data/rep.hpe
                      -undirected 1
                      -dimensions 128
                      -reg 0.01
                      -sample_times 5
                      -walk_steps 5
                      -negative_samples 5
                      -alpha 0.025
                      -threads 4
```
6. Transform the embedding file format into JSON format.
```bash
python3 ./script/rep_transform.py [-o OUTPUT_PATH] <representation_file_path>
```
7. Using the Jieba (Chinese word Segmentation) to get the segmentation for each event title.
```bash
python3 ./script/segement.py [-o OUTPUT_PATH] <title_file_path>
```
8. Using vector space model to retrieval top k similar (cosine similarity) training events' embeddings and take the average of them, then produce new embedding to the unseen event.
```
python3 ./src/vsm_propagation.py <unseen_event_file> 
                                 <embedding_JSON_file>
                                 <corpus_json_file>
                                 (title's segemntation)
```

## The Full Execution Flow for Propagation by Label-Embedding
The step1 to step7 are the same as the propagation by TFIDF.

8. Generate keywords form title and description for each event.
```
python3 ./script/textrank.py [-o OUTPUT]
                             [--textrank_word2vec TEXTRANK_WORD2VEC 1]
                             [--textrank_idf TEXTRANK_IDF 1]
                             <descripton_file_path>
```
9. Construct user-label(word) graph
```
python3 ./script/construct_user_word_graph.py [-o OUTPUT]
                                              <user_log event_keyword_json>
                                              <word_mapping_file>
```
11. Using proNet-core’s LINE-2nd to get the embeddings of the users and words(labels).
```
./proNet-core/cli/line -train ./data/user-label.data
                       -save ./data/textrank/rep.line2
                       -undirected 1
                       -order 2
                       -dimensions 128
                       -sample_times 4
                       -negative_samples 5
                       -alpha 0.025
                       -threads 4
```
13. Propagate HPE embedding based on events' aggregated label-embedding.
```
python3 ./src/label_propagation.py [--textrank_word2vec TEXTRANK_WORD2VEC 1]
                                   [--textrank_idf TEXTRANK_IDF 1]
                                   [--embedrank EMBEDRANK 1]
                                   <unseen_event_file>
                                   <embedding_file>
                                   <corpus_file>
                                   <concept_folder_path>
```
