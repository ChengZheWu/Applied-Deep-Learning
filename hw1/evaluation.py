import json
import pickle
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from slot_dataset import SeqClsDataset
from slot_model import SeqClassifier
from utils import Vocab
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
eval_file = "./data/slot/eval.json"
cache_dir = "./cache/slot/"
ckpt_dir = "/data/NFS/andy/course/ADL/hw1/slot_weights.pt"
max_len = 35

def cal_joint_acc(pred, label):
    pclass = pred.argmax(dim=2)
    correct = 0
    for i in range(len(label)):
        s_label = label[i][label[i]!=-100]
        length = len(s_label)
        if (s_label==pclass[i][:length]).all():
            correct += 1
    jacc = correct / len(label)
    return jacc

def cal_token_acc(pred, label):
    pclass = pred.argmax(dim=2)
    acc = 0
    for i in range(len(label)):
        s_label = label[i][label[i]!=-100]
        length = len(s_label)
        acc += (s_label==pclass[i][:length]).sum().item()/length
    acc /= len(label)
    return acc

def main():
    with open(cache_dir + "vocab.pkl", "rb") as f:
            vocab: Vocab = pickle.load(f)

    tag_idx_path = Path(cache_dir + "tag2idx.json")
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(Path(eval_file).read_text())
    dataset = SeqClsDataset(data, vocab, tag2idx, max_len)

    batch_size = 128
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(cache_dir + "embeddings.pt")

    model = SeqClassifier(embeddings=embeddings, hidden_size=256, num_layers=2, dropout=0.2, bidirectional=True, num_class=9)
    model.to(device)
    model.load_state_dict(torch.load(ckpt_dir))

    joint_acc = 0
    token_acc = 0
    all_pred = []
    all_label = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            data = batch[0].to(device)
            label = batch[1].to(device)
            pred = model(data)
            pclass = pred.argmax(dim=2)
            for i in range(len(label)):
                s_label = label[i][label[i]!=-100]
                length = len(s_label)
                sub_pred = []
                sub_label = []
                for j in range(length):
                    sub_pred.append(dataset.idx2label(pclass[i][j].item()))
                    sub_label.append(dataset.idx2label(label[i][j].item()))
                all_pred.append(sub_pred)
                all_label.append(sub_label)

            joint_acc += cal_joint_acc(pred, label)
            token_acc += cal_token_acc(pred, label)

    print("joint acc:", joint_acc/val_loader.__len__())
    print("token acc:", token_acc/val_loader.__len__())
    print(classification_report(all_label, all_pred, scheme=IOB2, mode="strict"))
    
if __name__ == "__main__":
    main()