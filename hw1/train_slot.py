import json
import pickle
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from slot_dataset import SeqClsDataset
from slot_model import SeqClassifier
from utils import Vocab

import matplotlib.pyplot as plt

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def cal_joint_acc(pred, label):
    pclass = pred.argmax(dim=2)
    correct = 0
    for i in range(len(label)):
        s_label = label[i][label[i] != -100]
        length = len(s_label)
        if (s_label == pclass[i][:length]).all():
            correct += 1
    jacc = correct / len(label)
    return jacc


def plot(train_metirc, val_metric, metric_name, loss=True):
    plt.plot(train_metirc, label='train_%s' % metric_name)
    plt.plot(val_metric, label="val_%s" % metric_name)
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

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size,
                              shuffle=True, collate_fn=datasets["train"].collate_fn)
    val_loader = DataLoader(datasets["eval"], batch_size=args.batch_size,
                            shuffle=False, collate_fn=datasets["eval"].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
                          bidirectional=args.bidirectional, num_class=9).to(device)
    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop - iterate over train dataloader and update model weights
    # Evaluation loop - calculate accuracy and save model weights
    his_trian_loss = []
    his_val_loss = []
    best_jacc = 0.7
    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()

        train_loss = 0
        train_jacc = 0
        train_len = 0
        val_loss = 0
        val_jacc = 0
        val_len = 0

        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data = batch[0].to(device)
            label = batch[1].to(device)
            pred = model(data)
            _pred = pred.permute(0, 2, 1)
            loss = criterion(_pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_jacc += cal_joint_acc(pred, label)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                data = batch[0].to(device)
                label = batch[1].to(device)
                pred = model(data)
                _pred = pred.permute(0, 2, 1)
                loss = criterion(_pred, label)
                val_loss += loss.item()
                val_jacc += cal_joint_acc(pred, label)

        his_trian_loss.append(train_loss/train_loader.__len__())
        his_val_loss.append(val_loss/val_loader.__len__())

        print('[%03d/%03d] %2.2f sec(s) Train Loss: %.4f Acc: %.4f| Val loss: %.4f Acc: %.4f' %
              (epoch + 1, args.num_epoch, time.time()-epoch_start_time,
               train_loss/train_loader.__len__(), train_jacc/train_loader.__len__(),
               val_loss/val_loader.__len__(), val_jacc/val_loader.__len__()))

        if val_jacc/val_loader.__len__() >= best_jacc:
            best_jacc = val_jacc/val_loader.__len__()
            torch.save(model.state_dict(), args.ckpt_dir / "weights.pt")
            print("saving model with acc:%.4f" % best_jacc)
    plot(his_trian_loss, his_val_loss, "loss")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=35)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
