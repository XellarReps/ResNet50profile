import os
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from mlperf_logging import mllog
from training import train

mllogger = mllog.get_mllogger()
mllog.config(
    filename=(os.getenv("LOG_FILE") or "resnet.log"),
    root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))))

DATA_PATH = os.getenv("DATASET_DIR")

def forward_log_in(num_layer):
    def hook(self, input):
        mllogger.event(key='start_' + self.__class__.__name__,
            value=num_layer)
    return hook


def forward_log_out(num_layer):
    def hook(self, input, output):
        torch.cuda.synchronize()
        mllogger.event(key='stop_' + self.__class__.__name__,
            value=num_layer)
    return hook

def main():
    parser = argparse.ArgumentParser(description="ResNet50 with profile")

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the ImageNet Object Localization Challenge dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=DATA_PATH, 
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = torchvision.models.resnet50(pretrained=True)
    model = model.to(device)

    for idx, layer in enumerate(model.modules()):
        layer.register_forward_pre_hook(forward_log_in(idx))
        layer.register_forward_hook(forward_log_out(idx))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, optimizer, criterion, train_loader, device, args)


if __name__ == '__main__':
    main()