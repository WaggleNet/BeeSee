import argparse
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import OistDataset
from model import ReducedUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args, model):
    dataset = OistDataset(args.data)
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

    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter()
    train_step = 0

    for epoch in range(args.epochs):
        for x, y in (pbar := tqdm(train_loader)):
            x = (x.float() / 255).to(DEVICE)
            y = (y.float() / 255).to(DEVICE)

            optim.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()

            pbar.set_description(f"Train: epoch={epoch} loss={loss.item():.3f}")
            writer.add_scalar("train_loss", loss.item(), train_step)
            train_step += 1

        with torch.no_grad():
            total_loss = 0
            for x, y in (pbar := tqdm(test_loader)):
                x = (x.float() / 255).to(DEVICE)
                y = (y.float() / 255).to(DEVICE)

                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item()

                pbar.set_description(f"Test: epoch={epoch} loss={loss.item():.3f}")

            total_loss /= len(test_loader)
            writer.add_scalar("test_loss", total_loss, train_step)

        torch.save(model.state_dict(), "model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("Training on device", DEVICE)

    model = ReducedUNet().to(DEVICE)

    train(args, model)


if __name__ == "__main__":
    main()
