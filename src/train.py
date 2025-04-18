import argparse
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from utils import *


def train(args, dataset, model):
    train_len = int(len(dataset) * 0.9)
    test_len = len(dataset) - train_len
    train_data, test_data = random_split(dataset, (train_len, test_len))
    loader_args = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        prefetch_factor=2,
    )
    train_loader = DataLoader(train_data, **loader_args)
    test_loader = DataLoader(test_data, **loader_args)

    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=str(args.logdir))
    train_step = 0

    for epoch in range(args.epochs):
        model.train()
        for x, y in (pbar := tqdm(train_loader)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x, logits=True)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

            pbar.set_description(f"Train: epoch={epoch} loss={loss.item():.3f}")
            writer.add_scalar("train_loss", loss.item(), train_step)
            train_step += 1

        with torch.no_grad():
            model.eval()
            total_loss = 0
            for x, y in (pbar := tqdm(test_loader)):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                pred = model(x, logits=True)
                loss = criterion(pred, y)
                total_loss += loss.item()

                pbar.set_description(f"Test: epoch={epoch} loss={loss.item():.3f}")

            total_loss /= len(test_loader)
            writer.add_scalar("test_loss", total_loss, train_step)
            writer.add_images("test_x", x, train_step)
            writer.add_images("test_y", y, train_step)
            writer.add_images("test_pred", pred, train_step)

        torch.save(model.state_dict(), args.logdir.name + ".pt")


def write_train_params(args, model):
    args.logdir.mkdir(exist_ok=True, parents=True)
    with open(args.logdir / "params.txt", "w") as f:
        attrs = ("data", "data_type", "resume", "model_type", "logdir", "batch_size", "epochs", "lr")
        for attr in attrs:
            print(f"{attr}: {getattr(args, attr)}", file=f)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num_params:", num_params, file=f)
        print("save_as:", args.logdir.name + ".pt", file=f)
        print(file=f)
        print(model, file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset directory.")
    parser.add_argument("--data_type", type=str, choices=DATA_CHOICES, required=True)
    parser.add_argument("--resume", type=Path, help="Model file to resume from.")
    parser.add_argument("--model_type", type=str, choices=MODEL_CHOICES, required=True)
    parser.add_argument("--logdir", type=Path, required=True, help="Path to tensorboard logs.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    print("Using device:", DEVICE)

    dataset_cls, model_cls, dataset_args = get_dataset_model(args.data_type, args.model_type)
    dataset = dataset_cls(dir=args.data, **dataset_args)
    model = model_cls().to(DEVICE)

    write_train_params(args, model)

    if args.resume is not None:
        print("Resuming from", args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))

    train(args, dataset, model)


if __name__ == "__main__":
    main()
