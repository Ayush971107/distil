import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SmallCIFAR10_CNN(nn.Module):
    def __init__(self):
        super(SmallCIFAR10_CNN, self).__init__()
        self.features = nn.Sequential(
            # Input: 3x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8x8x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4x128
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_data(batch_size=128):
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Just normalization for validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def train_kd(student, teacher, train_loader, epoch):
    student.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [KD-Train]", leave=False)
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        # ---------- forward ----------
        with torch.no_grad():                     # teacher prediction
            t_logits = teacher(inputs)

        s_logits = student(inputs)

        # ---------- losses ----------
        # hard-label loss
        ce_loss = ce_criterion(s_logits, targets)

        # soft-target loss (KL between softened distributions)
        s_log_prob = F.log_softmax(s_logits / TEMPERATURE, dim=1)
        t_prob     = F.softmax   (t_logits / TEMPERATURE, dim=1)
        kd_loss = kl_criterion(s_log_prob, t_prob) * (TEMPERATURE ** 2) # multiplying to scale it back according to ce loss

        loss = ALPHA * kd_loss + (1.0 - ALPHA) * ce_loss

        # ---------- backward ----------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = s_logits.max(1)
        total   += targets.size(0)
        correct += pred.eq(targets).sum().item()
        running_loss += loss.item()

        loop.set_postfix(loss=running_loss/(loop.n+1),
                         acc=100.*correct/total)

    return running_loss/len(train_loader), 100.*correct/total


