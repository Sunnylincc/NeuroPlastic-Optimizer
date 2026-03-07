from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def _build_synthetic_loader(batch_size: int, num_workers: int):
    train_x = torch.randn(1024, 1, 28, 28)
    train_y = torch.randint(0, 10, (1024,))
    test_x = torch.randn(256, 1, 28, 28)
    test_y = torch.randint(0, 10, (256,))
    train = TensorDataset(train_x, train_y)
    test = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def build_dataloaders(dataset: str, batch_size: int, num_workers: int):
    dataset = dataset.lower()
    if dataset == "synthetic_mnist":
        return _build_synthetic_loader(batch_size=batch_size, num_workers=num_workers)
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
