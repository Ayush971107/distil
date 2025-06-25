import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from teacher import Teacher
# Set random seed for reproducibility
torch.manual_seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Student(nn.Module):
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

def train_kd(student, teacher, train_loader, optimizer, ce_criterion, kl_criterion, temperature, alpha, epoch):
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
        s_log_prob = F.log_softmax(s_logits / temperature, dim=1)
        t_prob     = F.softmax   (t_logits / temperature, dim=1)
        kd_loss = kl_criterion(s_log_prob, t_prob) * (temperature ** 2) # multiplying to scale it back according to ce loss

        loss = alpha * kd_loss + (1.0 - alpha) * ce_loss

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

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item()
            
            progress_bar.set_postfix(loss=running_loss/(progress_bar.n+1), 
                                   acc=100.*correct/total)
    
    return running_loss/len(test_loader), 100.*correct/total

def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    num_epochs = 20

    # --- KD hyper-parameters  ----------------------------------------------------
    temperature = 4.0          # τ in the paper
    alpha       = 0.9          # weight on soft-target (KD) loss
    # ---------------------------------------------------------------------------
    
    # Initialize loss functions
    ce_criterion = nn.CrossEntropyLoss()                     # hard labels
    kl_criterion = nn.KLDivLoss(reduction='batchmean')       # soft targets (KL)
    
    # Load teacher model
    teacher = Teacher().to(device)
    try:
        teacher.load_state_dict(torch.load("teacher_best.pth", map_location=device))
        teacher.eval()  # inference-only
        for p in teacher.parameters():  # freeze weights
            p.requires_grad = False
        print("Successfully loaded teacher model")
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        return
    
    # Load data
    train_loader, test_loader, classes = load_data(batch_size)
    
    # Initialize model
    model = Student().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params:,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                         momentum=momentum, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    
    # Training loop
    best_acc = 0.0
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_kd(
            student=model,
            teacher=teacher,
            train_loader=train_loader,
            optimizer=optimizer,
            ce_criterion=ce_criterion,
            kl_criterion=kl_criterion,
            temperature=temperature,
            alpha=alpha,
            epoch=epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Update learning rate
        scheduler.step(test_acc)
        
        # Print epoch results
        print(f'Epoch: {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | ' \
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'distilled_model_best.pth')
            print(f'Best model saved with accuracy: {best_acc:.2f}%')
    
    print(f'Finished Training. Best test accuracy: {best_acc:.2f}%')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('small_cnn_training_history.png')
    plt.show()

if __name__ == '__main__':
    main()