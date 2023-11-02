import torchvision

# resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
resnet = torchvision.models.resnet18(pretrained=True)