from typing import List, Dict

from torch.utils.data import Dataset
import json as json
import torch
from torch.nn.utils.rnn import pad_sequence

class contextDataset(Dataset):
    def __init__(self, schema_path, data, mode, tokenizer, MAX_LENGTH=512):
        self.data = data
        self.mode = mode
        self.len = len(data)
        self.tokenizer = tokenizer
        self.max_len = MAX_LENGTH
        with open(schema_path, encoding="utf-8") as f:
            schemas = json.load(f)
        self.schema_dict = {}
        for schema in schemas:
            self.schema_dict[schema['service_name']]=[slot['name'] for slot in schema['slots']]
    def __getitem__(self, index):
        instance = self.data[index]
        return instance
        
    def __len__(self):
        return self.len

    def collate_fn(self, samples):
#         print('collate_fn samples:', samples)
        global input_ids_list, token_type_ids_list, attention_mask_list, label_list 
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = torch.zeros(len(samples),dtype = torch.long)
        for i, sample in enumerate(samples):
            question_list = []
            context_list = []

            for service in sample['services']:
                service_list = self.schema_dict[service]
                label = torch.zeros(len(service_list),dtype = torch.long)
                for slot in sample['slot']:
                    slot_service, slot_name = slot.split()
                    if slot_service == service and slot_name in service_list:
                        label[service_list.index(slot_name)] = 1
                label_list=torch.cat((label_list,label),0)
                question_list.extend(service_list)
                for slot in service_list:
                    context_list.append(sample['context'])
                
#                 for question in sample['paragraphs'] :
#                     question_list.append(sample['question'])
#                     contexts = self.context[context_id]

#                     if(sample['answers'][0]['start'] >= self.max_len-3-len(sample['question'])):
#                         contexts_len = self.max_len-3-len(sample['question'])
#                         contexts = contexts[(sample['answers'][0]['start'] - contexts_len//2):]
#                     contexts = contexts[:self.max_len-3-len(sample['question'])]
#                     context_list.append(contexts)
#     #                 print('question len:',len(sample['question']),'context len:',len(contexts))
#                 print(f'question_list:{len(question_list)},context_list:{len(context_list)}')
            encoding = self.tokenizer(question_list, context_list, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)

            input_ids = encoding['input_ids']
            token_type_ids = encoding['token_type_ids']
            attention_mask = encoding['attention_mask']
            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)
#             label_list.append(label)
        input_ids_list = pad_sequence(input_ids_list,batch_first=True)
        token_type_ids_list = pad_sequence(token_type_ids_list,batch_first=True)
        attention_mask_list = pad_sequence(attention_mask_list,batch_first=True)
#         label_list = pad_sequence(label_list,batch_first=True)
        return input_ids_list, token_type_ids_list, attention_mask_list, label_list