# NetworkTools

NetworkTools is a toolbox used for preparing data to conduct some experiments on [proNet-core](https://github.com/cnclabs/proNet-core)'s algorithm.

NetworkTools includes functions below:
- extract data which are written in csv format
- convert user-item record to user-list(of items) record
- split all data into a training set and a testing set
- export [proNet-core's training data format](https://github.com/cnclabs/proNet-core#task)

## The Full Execution Flow
1. Execute extact.sh will produce a user-item pairs data.
(The default filename of output is "user-item.data")
```bash
./extract.sh <csv_flile_path> <user_column_num> <item_column_num>
```
2. Execute generate_item_list.py will convert user-item pairs data into user-list(of items) pairs.
(The default filename of output is "itemsList.data")
```bash
python3 generate_item_list.py [-o OUTPUT_PATH] <user-item_file_path>
```
3. Execute data_split.py will produce a training set and a testing set, both of them come from the output file of the previous step. 
(The default filename of outputs are "training.data" and "testing.data")
```bash
python3 data_split.py [-o1 TRAIN_OUTPUT] [-o2 TEST_OUTPUT] <items-list_file_path>
```
4. Execute export.py will convert user-list(of items) data into the output data in [proNet-core's format](https://github.com/cnclabs/proNet-core#task).
(The default filename of output is "export.data")
```bash
python3 export.py [-o OUTPUT_PATH] <user-itemslist_file_path>
```
5. Using proNet-core's model to get the embeddings of the users and items.
```bash
./proNet-core/cli/hpe -train ./data/export.data -save ./data/rep.hpe -undirected 1 -dimensions 128 -reg 0.01 -sample_times 5 -walk_steps 5 -negative_samples 5 -alpha 0.025 -threads 4
```
6. Transform the embedding file format into JSON format.
```bash
python3 rep_transform.py [-o OUTPUT_PATH] <representation_file_path>
```
7. Using the Jieba (Chinese word Segmentation) to get the segmentation for each event title.
```bash
python3 segement.py [-o OUTPUT_PATH] <title_file_path>
```
8. Using vector space model to retrieval top k similar (cosine similarity) training events' embeddings and take the average of these, then produce new embedding to the unseen event.
