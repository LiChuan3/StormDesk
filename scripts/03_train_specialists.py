"""Train the CLIPER baseline and the GRU/Transformer DL specialists.

Strict year split: train 1980-2015, val 2016-2017. Saves models to
<work>/models/. Run on any GPU node (small models, minutes).
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.cliper import Cliper
from stormdesk.guidance import dl_models as dm


def load_cases(split):
    with open(os.path.join(work_dir("cases"), f"{split}.pkl"), "rb") as f:
        return pickle.load(f)


def build_tensors(cases):
    seqs, stats, ys = [], [], []
    for _, r in cases.iterrows():
        t = dm.case_tensors(r["history"], r["lat"], r["lon"], r["vmax"],
                            r["basin"], r["init"])
        if t is None:
            continue
        y = dm.targets_from_case(r)
        if not np.isfinite(y).any():
            continue
        seqs.append(t[0])
        stats.append(t[1])
        ys.append(y)
    return (torch.from_numpy(np.stack(seqs)), torch.from_numpy(np.stack(stats)),
            torch.from_numpy(np.stack(ys)))


def train_one(name, model, train_dl, val_t, device, epochs=40, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = float("inf")
    vs, vst, vy = (x.to(device) for x in val_t)
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for seq, stat, y in train_dl:
            seq, stat, y = seq.to(device), stat.to(device), y.to(device)
            opt.zero_grad()
            loss = dm.masked_huber(model(seq, stat), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(seq)
            n += len(seq)
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = float(dm.masked_huber(model(vs, vst), vy))
        if vloss < best:
            best = vloss
            dm.save_model(model, name)
        print(f"[{name}] epoch {ep+1}/{epochs} train {tot/n:.4f} val {vloss:.4f} best {best:.4f}",
              flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--only", choices=["cliper", "gru", "transformer"], default=None)
    args = ap.parse_args()

    train, val = load_cases("train"), load_cases("val")
    print(f"train {len(train)} cases, val {len(val)} cases")

    if args.only in (None, "cliper"):
        cl = Cliper().fit(train)
        cl.save()
        print("cliper saved")

    if args.only in (None, "gru", "transformer"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tt = build_tensors(train)
        vt = build_tensors(val)
        print(f"tensors: train {tt[0].shape}, val {vt[0].shape}")
        dl = DataLoader(TensorDataset(*tt), batch_size=args.bs, shuffle=True,
                        drop_last=True)
        if args.only in (None, "gru"):
            train_one("gru", dm.GRUSpecialist(), dl, vt, device, args.epochs)
        if args.only in (None, "transformer"):
            train_one("transformer", dm.TransformerSpecialist(), dl, vt, device, args.epochs)


if __name__ == "__main__":
    main()
