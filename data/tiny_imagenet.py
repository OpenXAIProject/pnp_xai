import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder('~/.cache/torch/tiny-imagenet-200/train', transform=data_transforms)
test_data = datasets.ImageFolder('~/.cache/torch/tiny-imagenet-200/val', transform=data_transforms)

loaders = {
    'train': torch.utils.data.DataLoader(
        train_data, 
        batch_size=128, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    ),
    'test': torch.utils.data.DataLoader(
        test_data, 
        batch_size=128, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    ),
}