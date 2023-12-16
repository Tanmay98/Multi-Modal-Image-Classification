import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

def build_dataset(is_train):
    # dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=ToTensor())
    dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]))
    torch.manual_seed(43)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    if is_train:
        return train_ds
    else:
        return val_ds
    
def get_text_embed(is_train, index):
    
    if is_train:
        text_embedding = torch.load(f"/home/tbaweja/811/text_embeddings_train/{index}.pt").float()
        return text_embedding
    else:
        text_embedding = torch.load(f"/home/tbaweja/811/text_embeddings_val/{int(index)}.pt").float()
        return text_embedding

class ImageDataset(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        self.dataset = build_dataset(is_train)
        # assert len(self.train_dataset) == len(self.train_text_embeddings)

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        text_emb = get_text_embed(self.is_train, index)
        image, class_label = self.dataset[index]
        image = image.permute(1,2,0).numpy()
        
        for _ in range(3):
            image = cv2.pyrUp(image)

        image = torch.from_numpy(image).permute(2,0,1)
        return image, class_label, text_emb