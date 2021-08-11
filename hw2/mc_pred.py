import os
import json
import transformers
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from eval import *
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union

data_path = "./dataset/private.json"
context_path = "./dataset/context.json"
output_path = "./mc_prediction.json"
weights_path = "./mc_weights/"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_checkpoint = "bert-base-chinese"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

def to_same_paragraphs(data):
    for i in range(len(data["paragraphs"])):
        while len(data["paragraphs"][i]) != 7:
            data["paragraphs"][i].append("")
    return data

def data_transfer(data, context):
    new_data = {"question":[], "paragraphs":[]}
    for sub in data:
        new_data["question"].append(sub["question"])
        new_data["paragraphs"].append([context[i] for i in sub["paragraphs"]])
    
    new_data = to_same_paragraphs(new_data)
    return new_data

def preprocess_function(examples):
    question = [[q] * 7 for q in examples["question"]]
    sub_contexts = []
    for i in range(len(examples["paragraphs"])):
        for p in examples["paragraphs"][i]:
            sub_contexts.append([p])

    question = sum(question, [])
    sub_contexts = sum(sub_contexts, [])
    
    max_length = 512
    tokenized_examples = tokenizer(question, sub_contexts, max_length=max_length, truncation=True)
    return {k: [v[i:i+7] for i in range(0, len(v), 7)] for k, v in tokenized_examples.items()}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = []
        for i, feature in enumerate(features):
            num_choices = len(features[i]["input_ids"])
            f = []
            for j in range(num_choices):
                d = {}
                for k, v in feature.items():
                    d.update({k: v[j]})
                f.append(d)
            flattened_features.append(f)
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        return batch

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

def main():
    private_data = load_json(data_path)
    context = load_json(context_path)

    transfer_data = data_transfer(private_data, context)
    datasets = Dataset.from_dict(transfer_data)

    encoded_datasets = datasets.map(preprocess_function, batched=True, batch_size=1000)
    
    model = transformers.AutoModelForMultipleChoice.from_pretrained(weights_path)
    
    trainer = transformers.Trainer(
    model,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)
    
    preds = trainer.predict(encoded_datasets)
    preds_class = np.argmax(preds[0], axis=1)
    
    with open(output_path,"w") as f:
        preds_list = []
        for i, sub in enumerate(private_data):
            sub_dict = {}
            sub_dict["id"] = sub["id"]
            sub_dict["question"] = sub["question"]
            sub_dict["relevant"] = sub["paragraphs"][preds_class[i]]
            preds_list.append(sub_dict)
        json.dump(preds_list, f) 
    
if __name__ == "__main__":
    main()
    