# Homework2

## How to run code

```shell
# download the weights
bash download.sh

#unzip the zip file
unzip ./mc_weights.zip
unzip ./qa_weights.zip
```

all data need to be in the dataset file
ex: dataset/context.json
            private.json

### Multiple Choice
```shell
#predict the mc_prediction.json
bash mc_pred.sh
```

### Question Answering
```shell
#predict the qa_prediction.json
bash run.sh
```

## Reference
https://github.com/huggingface/transformers/tree/master/examples  
https://huggingface.co/models?search=chinese
