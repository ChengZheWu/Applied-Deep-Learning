import json
# test_path = test_seen or test_unseen
def make_test_data(test_path):
    data_num = 16 if test_path == 'test_seen' else 5
    import json
    from pathlib import Path
    train_list = []
    data_train = []
    for idx in range(1, data_num):
        train_path = Path(f'../../data/{test_path}/dialogues_{idx:0>3}.json')
        data_train.extend(json.load(open(train_path)))

    for i, _ in enumerate(data_train):
        train_dict={}
        train_dict['context']=''
        train_dict['dialogue_id'] = data_train[i]['dialogue_id']
        train_dict['services'] = data_train[i]['services']
        for turn in data_train[i]['turns']:
            train_dict['context']+=turn['utterance']+' '
        train_list.append(train_dict)

    import torch
    schema_path = '../../data/schema.json'
    with open(schema_path, encoding="utf-8") as f:
        schemas = json.load(f)
    schema_dict = {}
    for schema in schemas:
        schema_dict[schema['service_name']]=[f"{schema['service_name']} {slot['name']}: {slot['description']}" for slot in schema['slots']]
    question_list = []
    context_list = []
    id_list = []
    label_list = []
    for i, sample in enumerate(train_list):
        for service in sample['services']:
            service_list = schema_dict[service]
    #                 label[service_list.index(key)] = 1
    #         label_list.extend(label)
            question_list.extend(service_list)

            for slot in service_list:
                context_list.append(sample['context'])
                id_list.append(sample['dialogue_id'])

    test_list = [id_list,question_list,context_list]
    return test_list

test_seen_list = make_test_data('test_seen')
test_unseen_list = make_test_data('test_unseen')

# load dataset
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
test_seen_data = {'idx':[], 'sentence1': [], 'sentence2': []}
test_unseen_data = {'idx':[], 'sentence1': [], 'sentence2': []}
test_seen_data['idx'] = test_seen_list[0]
test_seen_data['sentence1'] = [(test_seen_list[1][i]) for i, _ in enumerate(test_seen_list[0])]
test_seen_data['sentence2'] = [(test_seen_list[2][i]) for i, _ in enumerate(test_seen_list[0])]
test_unseen_data['idx'] = test_unseen_list[0]
test_unseen_data['sentence1'] =[test_unseen_list[1][i] for i, _ in enumerate (test_unseen_list[0])]
test_unseen_data['sentence2'] = [test_unseen_list[2][i] for i, _ in enumerate(test_unseen_list[0])]
test_seen_dataset = DatasetDict({"test":Dataset.from_dict(test_seen_data)})
test_unseen_dataset = DatasetDict({"test":Dataset.from_dict(test_unseen_data)})                             

# load model& tokenizer 
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, BertTokenizer
weights_path = "./weights"
tokenizer_path = "./tokenizer"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(weights_path, num_labels=2)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    correct = 0
    for i in range(len(labels)):
        if preds[i] == labels[i]:
            correct += 1
    acc = correct / len(labels)
    return {'accuracy': acc}

from transformers import TrainingArguments
learning_rate = 2e-5
batch_size = 10
epochs = 2
args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
#     fp16 = True,
#     metric_for_best_model="accuracy",
)

from transformers import Trainer
trainer = Trainer(
    model,
    args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

import numpy as np
mode = 'test_seen'
dataset  = test_seen_dataset if mode == 'test_seen' else test_unseen_dataset
print(f'encode dataset: {mode}')
encoded_dataset = dataset.map(preprocess_function, batched=True)

print(f'predict mode: {mode}')
preds = trainer.predict(encoded_dataset['test'])
preds_class = np.argmax(preds[0], axis=1)

output_path = f'../../data/{mode}.json'
with open(output_path,"w") as f:
    preds_list = []
    for i, dialogue_id in enumerate(test_seen_list[0]):
        if preds_class[i] != 0 :
            pred_string = ''
            pred_string = f'{dialogue_id}-{test_seen_list[1][i].split()[0]}-{test_seen_list[1][i].split()[1][:-1]}'
            preds_list.append(pred_string)
            print(pred_string)
    json.dump(preds_list, f) 
    print(f'predict finished: data/{mode}.json')
    
import numpy as np
mode = 'test_unseen'
dataset  = test_seen_dataset if mode == 'test_seen' else test_unseen_dataset
print(f'encode dataset: {mode}')
encoded_dataset = dataset.map(preprocess_function, batched=True)

print(f'predict mode: {mode}')
preds = trainer.predict(encoded_dataset['test'])
preds_class = np.argmax(preds[0], axis=1)

output_path = f'../../data/{mode}.json'
with open(output_path,"w") as f:
    preds_list = []
    for i, dialogue_id in enumerate(test_unseen_list[0]):
        if preds_class[i] != 0 :
            pred_string = ''
            pred_string = f'{dialogue_id}-{test_unseen_list[1][i].split()[0]}-{test_unseen_list[1][i].split()[1][:-1]}'
            preds_list.append(pred_string)
            print(pred_string)
    json.dump(preds_list, f)
    print(f'predict finished: data/{mode}.json')