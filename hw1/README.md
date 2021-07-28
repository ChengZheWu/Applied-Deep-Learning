# Intent Classification & Slot Tagging

## Model  
![model](https://github.com/ChengZheWu/Applied-Deep-Learning/blob/main/hw1/model.png)  

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
