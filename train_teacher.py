import torch
import torch.nn as nn
import torch.optim as optim
from base_template import Teacher, load_data, train_standard, test, plot_training_history, count_parameters, device


def main():
    print("=" * 50)
    print("TRAINING TEACHER MODEL")
    print("=" * 50)
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    num_epochs = 20
    
    # Load data
    train_loader, test_loader, classes = load_data(batch_size)
    print(f"Dataset loaded: {len(classes)} classes")
    
    # Initialize teacher model
    teacher = Teacher().to(device)
    
    # Count and display parameters
    total_params = count_parameters(teacher)
    print(f'Teacher model - Total trainable parameters: {total_params:,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(teacher.parameters(), lr=learning_rate, 
                         momentum=momentum, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)
    
    # Training loop
    best_acc = 0.0
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    
    print("\nStarting teacher training...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_standard(teacher, train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        test_loss, test_acc = test(teacher, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(teacher.state_dict(), 'teacher_best.pth')
            print(f"✓ New best model saved! Accuracy: {best_acc:.2f}%")
        
        print(f'Epoch: {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | ' \
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%')
    
    print("\n" + "=" * 50)
    print(f'✓ Teacher training completed!')
    print(f'✓ Best test accuracy: {best_acc:.2f}%')
    print(f'✓ Model saved as: teacher_best.pth')
    print("=" * 50)
    
    plot_training_history(train_losses, train_accs, test_losses, test_accs, 'teacher_training_history.png')
    print("✓ Training history plot saved as: teacher_training_history.png")


if __name__ == '__main__':
    main()
