import os
from dataclasses import dataclass
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import set_seed
from pnp_xai.pnp_xai import PnPXAI

import pdb


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

@dataclass
class UserInput:
    model: nn.Module
    data: torch.utils.data.DataLoader
    question: str
    task: str
    
    def __pos_init__(self):
        valid_questions = ['why', 'how']
        valid_tasks = ['image', 'tabular']
        
        if self.question not in valid_questions:
            raise ValueError(f"Invalid question: {self.question}. Choose from {', '.join(valid_questions)}.")
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task: {self.task}. Choose from {', '.join(valid_tasks)}.")

set_seed(seed=0)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


#-----------------------------------------------------------------------------#
#----------------------------------- model -----------------------------------#
#-----------------------------------------------------------------------------#

model = torchvision.models.resnet18(pretrained=True).to(device)


#-----------------------------------------------------------------------------#
#------------------------------------ data -----------------------------------#
#-----------------------------------------------------------------------------#

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'samples/')
        self.label_dir = os.path.join(self.root_dir, 'imagenet_class_index.json')
        
        with open(self.label_dir) as json_data:
            self.idx_to_labels = json.load(json_data)
        
        self.img_names = os.listdir(self.img_dir)
        self.img_names.sort()
        
        self.transform = transform
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = idx
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def idx_to_label(self, idx):
        return self.idx_to_labels[str(idx)][1]

class ToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, tensor):
        return tensor.to(self.device)

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ToDevice(device),
])

imagenet_data = ImageNetDataset(root_dir='./data/ImageNet/', transform=data_transforms)
loaders = {
    'test': torch.utils.data.DataLoader(
        imagenet_data, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    ),
}


#-----------------------------------------------------------------------------#
#---------------------------------- explain ----------------------------------#
#-----------------------------------------------------------------------------#

user_input = UserInput(
    model=model,
    data=loaders['test'],
    question='why',
    task='image',
)

# pdb.set_trace()

pnp_xai = PnPXAI()
pnp_xai.explain(user_input, savedir='./results/resnet18/ImageNet', sample_idx=[1, 14, 153, 216, 388, 440, 454, 656, 874, 933, 951, 966, 985, 999])
