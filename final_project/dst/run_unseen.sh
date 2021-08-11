# need to put unseen data under data/test_unseen/
cd 'text_classification'
python test_unseen.py
cd '../QA final'

python mktestdataset.py --input ../../data/test_unseen.json 
python QAtest.py --validation_file ./test_unseen.csv --output_dir ./output_unseen/
cp output_unseen/eval_predictions.json ./output_unseen.json
python remakeoutput.py --input ./output_unseen.json 
python remakesubmission.py --input ./kaggle_unseen.csv 