# Intent Classification & Slot Tagging

## Model  
![model](https://github.com/ChengZheWu/Applied-Deep-Learning/blob/main/hw1/model.png)  

## Results
Task (Metric %)             |Train  |Val    |Public |Private|Remark
:--------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:
Intent Classification (ACC) |99.80  |93.29  |92.400 |92.755 |
Intent Classification (ACC) |99.73  |93.66  |92.888 |93.377 |no relu
Slot Tagging (Joint ACC)    |91.17  |78.61  |71.903 |73.847 |
Slot Tagging (Joint ACC)    |91.67  |80.07  |75.656 |77.170 |no relu

 x            |precision |recall    |F1-score  |support
:------------:|:--------:|:--------:|:--------:|:--------:
date          |0.75      |0.74      |0.74      |206
first name    |0.93      |0.88      |0.90      |102
last name     |0.89      |0.73      |0.80      |78
people        |0.77      |0.71      |0.74      |238
time          |0.84      |0.90      |0.87      |218
micro avg     |0.81      |0.79      |0.80      |842
macro avg     |0.84      |0.79      |0.81      |842
weighted avg  |0.81      |0.79      |0.80      |842

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Reproducing
### file position  
#### data:  
./data/intent or slot/train.json, test.json, eval.json  
#### preprocessing data:  
./cache/intent or slot/...  
#### model weights:  
./ckpt/inten or slot/...  
#### prediction csv file:  
./...  
```shell
# download the model weights from https://drive.google.com/drive/folders/1UagyKmQL69jR43DCu9gBNsZdzWqkgKjm?usp=sharing
# or run
bash download.sh

# predict the results
bash intent_cls.sh
bash slot_tag.sh
```

## Training, Prediction, and Evaluation
```shell
# train intent model
python train_intent.py
# predict intent model
python test_intent.py

# train slot model
python train_slot.py
# predict slot model
python test_slot.py

# evaluate the slot model on validation dataset
python evaluate.py
```
