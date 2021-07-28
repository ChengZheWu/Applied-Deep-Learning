import json
import pickle
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from intent_dataset import SeqClsDataset
from intent_model import SeqClassifier
from utils import Vocab

import matplotlib.pyplot as plt

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def plot(train_metirc, val_metric, metric_name, loss=True):
    plt.plot(train_metirc, label='train_%s' %metric_name)
    plt.plot(val_metric, label="val_%s" %metric_name)
    plt.xlabel('epochs')
    plt.ylabel(metric_name)
    if loss:
        plt.legend(loc='upper right')
    else:
        plt.legend(loc='lower right')
    plt.show()


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    
    # crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=datasets["train"].collate_fn)
    val_loader = DataLoader(datasets["eval"], batch_size=args.batch_size, shuffle=False, collate_fn=datasets["eval"].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    # init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
                          bidirectional=args.bidirectional, num_class=150).to(device)
    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop - iterate over train dataloader and update model weights
    # Evaluation loop - calculate accuracy and save model weights
    his_trian_loss = []
    his_val_loss = []
    best_acc = 0.9
    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()

        train_loss = 0
        train_acc = 0
        train_len = 0
        val_loss = 0
        val_acc = 0
        val_len = 0
        
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data = batch[0].to(device)
            label = batch[1].to(device)
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (pred.argmax(dim=1) == label).sum().item()/len(data)
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                data = batch[0].to(device)
                label = batch[1].to(device)
                pred = model(data)
                loss = criterion(pred, label)
                val_loss += loss.item()
                val_acc += (pred.argmax(dim=1) == label).sum().item()/len(data)
                
        his_trian_loss.append(train_loss/train_loader.__len__())
        his_val_loss.append(val_loss/val_loader.__len__())

        print('[%03d/%03d] %2.2f sec(s) Train Loss: %.4f Acc: %.4f| Val loss: %.4f Acc: %.4f' % \
                (epoch + 1, args.num_epoch, time.time()-epoch_start_time, \
                 train_loss/train_loader.__len__(), train_acc/train_loader.__len__(), \
                 val_loss/val_loader.__len__(), val_acc/val_loader.__len__()))

        if val_acc/val_loader.__len__() >= best_acc:
            best_acc = val_acc/val_loader.__len__()
            torch.save(model.state_dict(), args.ckpt_dir / "weights.pt")
            print("saving model with acc:%.4f" %(best_acc))
    plot(his_trian_loss, his_val_loss, "loss")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=28)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    # training
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
