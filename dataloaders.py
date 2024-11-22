import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from os.path import join
import random
from tqdm import tqdm

def load_train_embeddings(model_name = "dino-vits", num_views=1, file_name =  lambda i: f"embeddings_{i+1}.pt"):
    path = f"/datadrive/pytorch-data/imagenet-{model_name}"
    embeddings_list = []
    for i in tqdm(range(num_views)):
        embeds = torch.load(join(path, file_name(i)), map_location="cpu")
        embeddings_list.append(embeds)

    emb_train = torch.stack(embeddings_list, dim=1)
    label_train = torch.load(join(path, "train_labels.pt"), map_location="cpu").to(torch.int)
    return emb_train, label_train

def load_val_embeddings(model_name = "dino-vits"):
    path = f"/datadrive/pytorch-data/imagenet-{model_name}"
    emb_val = torch.load(join(path, "embeddings_val.pt"), map_location="cpu")
    targets_val = torch.load(
        join(path, "labels_val.pt"), map_location="cpu"
    ).to(torch.int)
    return emb_val, targets_val

class ContrastiveDatasetFromList(Dataset):
    def __init__(self, x_views, labels, num_views):
        self.x_v = x_views  # Shape: (b, num_views, d)
        self.labels = labels  # Shape: (b,)
        self.num_views = num_views
        self.augmentation_idx = list(range(x_views.shape[1]))  # List of views

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Sample random views for the batch
        views_idx = random.sample(self.augmentation_idx, self.num_views)
        views = self.x_v[idx, views_idx, :]  # Select views, shape: [num_views, d]
        
        views_reshaped = views.view(-1, views.shape[-1])  # Shape: [n_views, d]
        labels_repeated = self.labels[idx].repeat(self.num_views)  # Shape: [n_views]
        indices_repeated = torch.tensor([idx] * self.num_views)  # Shape: [n_views]

        return views_reshaped, labels_repeated, indices_repeated


class ContrastiveDatasetFromList(Dataset):
    def __init__(self, x_views, labels, num_views):
        self.x_v = x_views  # Shape: (b, num_views, d)
        self.labels = labels  # Shape: (b,)
        self.num_views = num_views
        self.augmentation_idx = list(range(x_views.shape[1]))  # List of views

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Sample random views for the batch
        views_idx = random.sample(self.augmentation_idx, self.num_views)
        views = self.x_v[idx, views_idx, :]  # Select views, shape: [num_views, d]
        
        views_reshaped = views.view(-1, views.shape[-1])  # Shape: [n_views, d]
        return views_reshaped, self.labels[idx], idx

class ContrastiveDatasetFromList(Dataset):
    def __init__(self, x_views, labels, num_views):
        self.x_v = x_views  # Shape: (b, num_views, d)
        self.labels = labels  # Shape: (b,)
        self.num_views = num_views
        self.augmentation_idx = list(range(x_views.shape[1]))  # List of views

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Sample random views for the batch
        views_idx = random.sample(self.augmentation_idx, self.num_views)
        views = self.x_v[idx, views_idx, :]  # Select views, shape: [num_views, d]
        
        views_reshaped = views.view(-1, views.shape[-1])  # Shape: [n_views, d]
        labels_repeated = self.labels[idx].repeat(self.num_views)  # Shape: [n_views]
        indices_repeated = torch.tensor([idx] * self.num_views)  # Shape: [n_views]

        return views_reshaped, labels_repeated, indices_repeated        
class ContrastiveDatasetFromListwKNN(Dataset):
    def __init__(self, x_views, labels, num_views, knn_indices, num_neighbors=2):
        self.x_v = x_views
        self.labels = labels
        self.num_views = num_views
        self.augmentation_idx = list(range(x_views.shape[1]))  # List of views
        self.knn_indices = knn_indices
        self.num_neighbors = num_neighbors  # Number of neighbors to sample

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Sample random views for the batch
        views_idx = random.sample(self.augmentation_idx, self.num_views)
        views = self.x_v[idx, views_idx, :]  # Select views, shape: [num_views, d]
        
        # Sample multiple neighbors
        if self.num_neighbors > self.knn_indices.shape[1]:
            neighbor_indices = random.choices(list(self.knn_indices[idx, :]), k=self.num_neighbors)
        else:
            neighbor_indices = random.sample(list(self.knn_indices[idx, :]), self.num_neighbors)
        knn_views = []
        
        for neighbor_idx in neighbor_indices:
            neighbor_view = random.randint(0, self.num_views-1)
            knn_views.append(self.x_v[neighbor_idx, neighbor_view, :])

        # Concatenate all views
        knn_views = torch.stack(knn_views, dim=0)  # Stack the neighbor views
        views = torch.cat((views, knn_views), dim=0)
        
        views_reshaped = views.view(-1, views.shape[-1])
        labels_repeated = torch.tensor([self.labels[idx]] * self.num_views + [self.labels[neighbor_idx] for neighbor_idx in neighbor_indices])
        indices_repeated = torch.tensor([idx] * self.num_views + neighbor_indices)
        
        return views_reshaped, labels_repeated, indices_repeated

def custom_collate_fn(batch):
    data, labels, target = zip(*batch)
    data = torch.stack(data)  # Shape: [b, n, ...] where ... could be [d] or [h, w] 
    # Determine the shape of the last dimensions
    b, n = data.shape[:2]
    remaining_dims = data.shape[2:]  # Shape: [d] for embeddings or [C, H, W] for images

    # Reshape data to [b * n, ...]
    data = data.view(b * n, *remaining_dims)  # Shape: [b * n, d] or [b * n, C, H, W]
    
    labels = torch.cat(labels)  # Shape: [b * n]
    target = torch.cat(target)  # Shape: [b * n]
    return data, labels, target

class ContrastiveDatasetFromImages(Dataset):
    def __init__(self, dataset, num_views, transform=None, distict_views = True):
        self.dataset = dataset  # CIFAR-10 or CIFAR-100 dataset
        self.num_views = num_views
        self.transform = transform  # Augmentations to apply
        self.distict_views = distict_views

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        views = [self.transform(img) for _ in range(self.num_views)]
        views_stacked = torch.stack(views, dim=0)
        if not self.distict_views:
            label = torch.tensor([label] * self.num_views)
            idx = torch.tensor([idx] * self.num_views)

        return views_stacked, label, idx

def get_contrastive_dataloaders(batch_size=256, num_views=2, dataset_name='cifar10', num_workers=24, size=32, root='./data'):
    """Create CIFAR-10/100 dataloaders for contrastive learning with raw images."""
    
    # Define mean and std based on dataset
    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Define the normalization transformation
    normalize = transforms.Normalize(mean=mean, std=std)

    # Define the full transformation pipeline for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    # Define the transformation pipeline for testing (no augmentations)
    test_transform = transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        normalize,
    ])

    # Load dataset
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=None)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=None)

    # Wrap dataset for contrastive learning
    train_contrastive_dataset = ContrastiveDatasetFromImages(train_dataset, num_views=num_views, transform=train_transform)
    test_contrastive_dataset = ContrastiveDatasetFromImages(test_dataset, num_views=num_views, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_contrastive_dataset,
                              batch_size=batch_size//num_views,
                              shuffle=True, 
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=custom_collate_fn,
                              pin_memory=False)

    test_loader = DataLoader(test_contrastive_dataset,
                             batch_size=batch_size//num_views,
                             shuffle=False, 
                             drop_last=False,
                             num_workers=num_workers,
                             collate_fn=custom_collate_fn,
                             pin_memory=False)

    return train_loader, test_loader


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index

def get_mnist_dataloaders(limit=5, batch_size=5000, data_dir='data'):
    # Define a transformation that includes flattening
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the tensor
    ])

    # Load the MNIST dataset with the transformation
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Vectorized filtering to keep only samples where the label is less than the limit
    train_indices = torch.nonzero(torch.tensor(mnist_train.targets) < limit).squeeze()
    test_indices = torch.nonzero(torch.tensor(mnist_test.targets) < limit).squeeze()

    # Create subsets of filtered indices
    mnist_train_filtered = Subset(mnist_train, train_indices)
    mnist_test_filtered = Subset(mnist_test, test_indices)

    # Wrap the filtered datasets with the IndexedDataset to include image_index
    mnist_train_indexed = IndexedDataset(mnist_train_filtered)
    mnist_test_indexed = IndexedDataset(mnist_test_filtered)

    # Create DataLoaders
    train_loader = DataLoader(mnist_train_indexed, batch_size=batch_size, shuffle=True, num_workers = 24)
    test_loader = DataLoader(mnist_test_indexed, batch_size=len(test_indices), shuffle=True, num_workers = 24)

    return train_loader, test_loader

class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    
    def __call__(self, x):
        return self.transform(x)

def get_cifar10_dataloaders(root, batch_size=256, num_views=2, num_workers=0):
    train_transform = SimCLRTransform()
    dataset = datasets.CIFAR10(root=root, train=True, download=True)
    
    train_dataset = ContrastiveDatasetFromImages(
        dataset=dataset,
        num_views=num_views,
        transform=train_transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        #collate_fn=custom_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    
    return train_loader