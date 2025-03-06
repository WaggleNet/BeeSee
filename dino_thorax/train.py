import argparse
from pathlib import Path

from tqdm import tqdm

import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from constants import *
from dataset import ThoraxDataset
from dataset_oist import OistDataset
from model import DinoSegmentation


def train(args, model):
    def prepare_batch(x, y):
        """
        Helper function to preprocess batch (and handle differences between datasets, etc.)
        """
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if args.dataset_type == "oist":
            x = torch.cat([x] * 3, dim=1)
            y = F.resize(y, (y.shape[2] // 14, y.shape[3] // 14))

        return x, y

    if args.dataset_type == "oist":
        dataset = OistDataset(args.data, output_res=448)
    elif args.dataset_type == "thorax":
        dataset = ThoraxDataset(args.data)
    else:
        raise ValueError("Invalid dataset type")
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
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.lr_decay)

    writer = SummaryWriter(log_dir=str(args.logdir))
    train_step = 0

    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(pbar := tqdm(train_loader)):
            x, y = prepare_batch(x, y)

            pred = model(x, logits=True)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_step += 1

            curr_lr = optim.param_groups[0]["lr"]
            pbar.set_description(f"Train: epoch={epoch} loss={loss.item():.3f} lr={curr_lr:.3e}")
            writer.add_scalar("train_loss", loss.item(), train_step)
            writer.add_scalar("lr", curr_lr, train_step)

        with torch.no_grad():
            total_loss = 0
            for x, y in (pbar := tqdm(test_loader)):
                x, y = prepare_batch(x, y)

                pred = model(x, logits=True)
                loss = criterion(pred, y)
                total_loss += loss.item()

                pbar.set_description(f"Test: epoch={epoch} loss={loss.item():.3f}")

            total_loss /= len(test_loader)
            writer.add_scalar("test_loss", total_loss, train_step)
            writer.add_images("test_x", x, train_step)
            writer.add_images("test_y", y, train_step)
            writer.add_images("test_pred", pred, train_step)

        torch.save(model.state_dict(), "model.pt")
        lr_scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset directory.")
    parser.add_argument("--dataset_type", type=str, choices=["oist", "thorax"], default="thorax")
    parser.add_argument("--resume", type=Path, help="Model file to resume from.")
    parser.add_argument("--logdir", type=Path, default="runs", help="Path to tensorboard logs.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.7)
    args = parser.parse_args()

    print("Training on device", DEVICE)

    model = DinoSegmentation().to(DEVICE)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", num_params)

    args.logdir.mkdir(exist_ok=True, parents=True)
    with open(args.logdir / "params.txt", "w") as f:
        print("Number of parameters in model:", num_params, file=f)
        print("Data:", args.data, file=f)
        print("Dataset type:", args.dataset_type, file=f)
        print("Resume from:", args.resume, file=f)
        print("Log dir:", args.logdir, file=f)
        print("Batch size:", args.batch_size, file=f)
        print("Epochs:", args.epochs, file=f)
        print("LR:", args.lr, file=f)
        print("LR decay:", args.lr_decay, file=f)
        print(model, file=f)

    if args.resume is None:
        pass
        # print("Using Xavier weight init.")
        # model.init_weights()
    else:
        print("Resuming from", args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))

    train(args, model)


if __name__ == "__main__":
    main()
