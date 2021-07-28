import json
import pickle
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from intent_dataset import SeqClsDataset
from intent_model import SeqClassifier
from utils import Vocab

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    print("load data and model")
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn_test)
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, 
                          bidirectional=args.bidirectional, num_class=dataset.num_classes).to(device)
    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))

    # predict dataset
    print("start predict...")
    all_pred = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            data = batch.to(device)
            pred = model(data)
            pclass = pred.argmax(dim=1)
            for c in pclass:
                all_pred.append(c.item())

    # write prediction to file (args.pred_file)
    f = open(args.pred_file, "w")
    writer = csv.writer(f)
    writer.writerow(["id", "intent"])
    for i in range(len(all_pred)):
        intent = dataset.idx2label(all_pred[i])
        writer.writerow(["test-%d" %i, intent])
    f.close()
    print("done")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/weights.pt"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=28)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
