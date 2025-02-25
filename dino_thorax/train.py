import argparse
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms as T
from torchvision.io import read_image

from model import DinoSegmentation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ThoraxDataset(Dataset):
    def __init__(self, data_dir: Path, res=448):
        assert res % 14 == 0

        self.data_dir = data_dir
        self.indices = set()
        for f in data_dir.iterdir():
            if "_img" in f.stem:
                self.indices.add(f.stem.split("_")[0])
        self.indices = list(self.indices)

        self.res = res

        self.transform_both = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
        ])
        self.transform_x = T.Compose([
            T.Resize((res, res)),
        ])
        self.transform_y = T.Compose([
            T.Resize((res // 14, res // 14)),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = read_image(str(self.data_dir / f"{self.indices[idx]}_img.jpg")).float() / 255
        label = read_image(str(self.data_dir / f"{self.indices[idx]}_label.jpg")).float() / 255

        both = torch.cat([img, label], dim=0)
        both = self.transform_both(both)
        img, label = both[:3], both[3:]
        img = self.transform_x(img)
        label = self.transform_y(label)

        return img, label


def train(args, model):
    dataset = ThoraxDataset(args.data)
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
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.lr_decay)

    writer = SummaryWriter(log_dir=str(args.logdir))
    train_step = 0

    for epoch in range(args.epochs):
        for x, y in (pbar := tqdm(train_loader)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optim.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()

            curr_lr = optim.param_groups[0]["lr"]
            pbar.set_description(f"Train: epoch={epoch} loss={loss.item():.3f} lr={curr_lr:.3e}")
            writer.add_scalar("train_loss", loss.item(), train_step)
            writer.add_scalar("lr", curr_lr, train_step)
            train_step += 1

        with torch.no_grad():
            total_loss = 0
            for x, y in (pbar := tqdm(test_loader)):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                pred = model(x)
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
    parser.add_argument("--resume", type=Path, help="Model file to resume from.")
    parser.add_argument("--logdir", type=Path, default="runs", help="Path to tensorboard logs.")
    parser.add_argument("--batch_size", type=int, default=64)
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