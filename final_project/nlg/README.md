# Final Project
# Task 2 NLG

## Environment
```shell
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
cd example/pytorch/text-generation
pip install -r requirements.txt
pip install parlai
```

## Download model weight
```shell
bash download.sh
```

## Chit Chat Generation
```shell
# predict chit chat (it will take a lot of time)
parlai interactive -mf ./train_90M < lm.input.test_seen.cc.txt > lm.output.test_seen.cc.txt
# data will save in current directory
```

## Arranger
```shell
# predict the weight for each chit chat and generate final nlg output
python nlg_output.py
# data will save in current directory
```