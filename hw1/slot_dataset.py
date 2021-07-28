import torch
from typing import List, Dict
from torch.utils.data import Dataset
from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        text_list = []
        label_list = []
        for s in samples:
            text_list.append(s["tokens"])
            sub_label = []
            for t in s["tags"]:
                sub_label.append(self.label2idx(t))
            while len(sub_label) < self.max_len:
                sub_label.append(-100)
            label_list.append(sub_label)
        text_list = self.vocab.encode_batch(text_list, self.max_len)
        text_list = torch.tensor(text_list, dtype=torch.int64)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        return text_list, label_list
    
    def collate_fn_test(self, samples: List[Dict]) -> Dict:
        text_list = []
        for s in samples:
            text_list.append(s["tokens"])
        text_list = self.vocab.encode_batch(text_list, self.max_len)
        text_list = torch.tensor(text_list, dtype=torch.int64)
        return text_list
        
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
