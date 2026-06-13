import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def build_dataset_cifar10(data_dir='/home/liujiang/projects/dl/model_re/data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def get_data_loaders(dataset_name="CIFAR10", batch_size=128, num_workers=2):
    if dataset_name == "CIFAR10":
        train_dataset, test_dataset = build_dataset_cifar10()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader
