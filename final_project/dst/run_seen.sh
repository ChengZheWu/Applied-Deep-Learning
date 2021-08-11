# # need to put seen data under data/test_seen/
cd 'text_classification'
python test_seen.py
cd '../QA final'

python mktestdataset.py --input ../../data/test_seen.json --seen True
python QAtest.py --validation_file ./test_seen.csv --output_dir ./output_seen/
cp output_seen/eval_predictions.json ./output_seen.json
python remakeoutput.py --input ./output_seen.json --seen True
python remakesubmission.py --input ./kaggle_seen.csv --seen True