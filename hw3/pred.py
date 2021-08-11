import os
import jsonlines
import nltk
import torch
import numpy as np
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import PreTrainedModel
from typing import Optional, Union
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from tw_rouge import get_rouge

# data_path = "./input.jsonl"
data_path = "./input.jsonl"
output_path = "./output.jsonl"
weights_path = "./weights"
model_checkpoint = "google/mt5-small"
nltk.download('punkt')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_jsonl(path):
    data = []
    with jsonlines.open(path, "r") as f:
        for row in f:
            data.append(row)
    return data

@dataclass
class DataCollatorForSeq2Seq:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return features

def main():
    print("data loading...")
    data = load_jsonl(data_path)

    test_data = {"maintext":[], "id":[]}
    for sub in data:
        test_data["maintext"].append(sub["maintext"])
        test_data["id"].append(sub["id"])

    test_datasets = Dataset.from_dict(test_data)
    
    print("data preprocessing...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    max_input_length = 1024
    max_target_length = 128

    def preprocess_function(examples):
        inputs = [doc for doc in examples["maintext"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        return model_inputs

    tokenized_datasets = test_datasets.map(preprocess_function, batched=True)
    
    print("model loading...")
    model = AutoModelForSeq2SeqLM.from_pretrained(weights_path)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    test_dataloader = trainer.get_test_dataloader(tokenized_datasets)
    
    max_target_length = 128
    pred_list=[]
    label_list=[]

    for i, batch in enumerate(test_dataloader):
        attention_mask = batch['attention_mask'].to(device)
        inputs = batch['input_ids'].to(device)
        preds = model.generate(
            input_ids=inputs, 
            attention_mask=attention_mask, 
            max_length=max_target_length,
            num_beams=2,
        )

        for p in preds.cpu().numpy():
            pred_list.append(p)

    decoded_preds = tokenizer.batch_decode(pred_list, skip_special_tokens=True)
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    
    with jsonlines.open(output_path, "w") as writer:
        for i in range(len(decoded_preds)):
            d = {"title":decoded_preds[i], "id":data[i]["id"]}
            writer.write(d)
            
if __name__ == "__main__":
    main()