
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import BloodMNIST, INFO
import matplotlib.pyplot as plt
import time
import os

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

def get_loader(split, batch_size=64):
    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = BloodMNIST(split=split, transform=data_transform, download=True)
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(split == 'train'))
    return loader

class Net(nn.Module):
    def __init__(self, use_softmax=False):
        super(Net, self).__init__()
        self.use_softmax = use_softmax
        
        # 1. Conv + ReLU + MaxPool
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Conv + ReLU + MaxPool
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. Conv + ReLU + MaxPool
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dimension calc:
        # 28x28 -> pool -> 14x14
        # 14x14 -> pool -> 7x7
        # 7x7   -> pool -> 3x3 (floor(7/2))
        self.flatten_size = 128 * 3 * 3 # = 1152
        
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) 
        
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        
        if self.use_softmax:
            x = torch.nn.functional.softmax(x, dim=1)
            
        return x

def train_model(use_softmax, epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}, Use Softmax: {use_softmax}")
    
    model = Net(use_softmax=use_softmax).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = get_loader('train', batch_size=64)
    val_loader = get_loader('val', batch_size=64)
    test_loader = get_loader('test', batch_size=64)
    
    train_losses = []
    val_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze().long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze().long()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        
        # Test
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze().long()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
        test_acc = 100 * test_correct / test_total
        test_accuracies.append(test_acc)
        
        epoch_duration = time.time() - epoch_start
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_duration:.2f}s")
            
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f}s")
    return train_losses, val_accuracies, test_accuracies

def plot_results(res_logits, res_softmax):
    epochs = range(1, 201)
    train_loss_logits, val_acc_logits, test_acc_logits = res_logits
    train_loss_softmax, val_acc_softmax, test_acc_softmax = res_softmax
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss_logits, label='Logits + MaxPool')
    plt.plot(epochs, train_loss_softmax, label='Softmax + MaxPool')
    plt.title('Training Loss (MaxPooling)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_acc_logits, label='Logits + MaxPool')
    plt.plot(epochs, val_acc_softmax, label='Softmax + MaxPool')
    plt.title('Validation Accuracy (MaxPooling)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_acc_logits, label='Logits + MaxPool')
    plt.plot(epochs, test_acc_softmax, label='Softmax + MaxPool')
    plt.title('Test Accuracy (MaxPooling)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results_q1_2.png')
    print("Plots saved to 'results_q1_2.png'")

if __name__ == '__main__':
    # Experiment 1: Without Softmax (Correct)
    print("--- Starting Q1.2 Experiment 1: Without Softmax (Logits) + MaxPool ---")
    results_logits = train_model(use_softmax=False, epochs=200)
    
    # Experiment 2: With Softmax (Incorrect)
    print("\n--- Starting Q1.2 Experiment 2: With Softmax + MaxPool ---")
    results_softmax = train_model(use_softmax=True, epochs=200)
    
    plot_results(results_logits, results_softmax)
