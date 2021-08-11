# Chinese Question Answering
This task can be decomposed into 2 tasks:
* Context selection: determine which context is relevant.
* Span selection: determine the start and end position of the answer span in the context.  
  
![pipeline](https://github.com/ChengZheWu/Applied-Deep-Learning/blob/main/hw2/pipeline.png)

## Multiple Choice
### Input: Question、Context list
Q: "最早的鼓可以追溯至什麼古文明?"  
1. "鼓是一種打擊樂器，...，最早的鼓出現於西元前六千年的兩河文明"  
2. "盧克萊修生於共和國末期，...，被古典主義文學視為經典"  
3. "視網膜又稱視衣，...，約3mm2大的橢圓。"  
### Output: Category
1. "鼓是一種打擊樂器，...，最早的鼓出現於西元前六千年的兩河文明"  
## Question Answering
### Input: Question、Context
Q: "最早的鼓可以追溯至什麼古文明?"  
1. "鼓是一種打擊樂器，...，最早的鼓出現於西元前六千年的兩河文明"  
### Output: Answer
"兩河文明"  

## Model
Huggingface Transformers中提供許多pretrained transformer-base model讓使用者做finetune，像是BERT等model，因此本次作業也使用Huggingface裡面的model來訓練。  
### Multiple Choice
1. BERT (“bert-base-chinese”)
### Question Answering
1. BERT (“bert-base-chinese”)
2. MacBERT ("hfl/chinese-macbert-large")

## Results
### Multiple Choice
Model |ACC(%) | 
:----:|:-----:|
BERT  |94.2   |

### Question Answering
Model   |EM(%)  |F1(%)  | 
:------:|:-----:|:-----:|
BERT    |78.9   |86.7
MaxBERT |85.5   |91.6

### Pretrained vs Not Pretrained
BERT           |EM(%)  |F1(%)  | 
:-------------:|:-----:|:-----:|
pretrained     |78.9   |86.7
not pretrained |3.5    |5.9

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
