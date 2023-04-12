import torch
import torchvision
import torchvision.transforms as transforms

def train(model, optimizer, criterion, data_loader, device, args):
    for epoch in range(args.epochs):
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}')
    print(f'Finished Training, Loss: {loss.item():.4f}')