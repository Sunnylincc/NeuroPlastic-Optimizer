from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloaders(dataset: str, batch_size: int, num_workers: int):
    dataset = dataset.lower()
    if dataset in {"mnist", "fashionmnist"}:
        normalize = transforms.Normalize((0.5,), (0.5,))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        dset_cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        train = dset_cls("data", train=True, download=True, transform=transform)
        test = dset_cls("data", train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
            ]
        )
        train = datasets.CIFAR10("data", train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10("data", train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
