import torch
import torch.nn as nn
import torch.optim as optim
from base_template import Teacher, Student, load_data, train_kd, test, plot_training_history, count_parameters, device


def load_pretrained_teacher(model_path='teacher_best.pth'):
    teacher = Teacher().to(device)
    try:
        teacher.load_state_dict(torch.load(model_path, map_location=device))
        teacher.eval()
        print(f"✓ Loaded pre-trained teacher model from: {model_path}")
        return teacher
    except FileNotFoundError:
        print(f"✗ Error: Teacher model not found at {model_path}")
        print("Please train the teacher model first using train_teacher.py")
        return None


def main():
    print("=" * 50)
    print("TRAINING STUDENT MODEL (Knowledge Distillation)")
    print("=" * 50)

    teacher = load_pretrained_teacher()
    if teacher is None:
        return
    
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 25
    
    temperature = 4.0
    alpha = 0.9
    
    train_loader, test_loader, classes = load_data(batch_size)
    print(f"Dataset loaded: {len(classes)} classes")
    
    student = Student().to(device)
    
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    compression_ratio = teacher_params / student_params
    
    print(f'Teacher model - Trainable parameters: {teacher_params:,}')
    print(f'Student model - Trainable parameters: {student_params:,}')
    print(f'Compression ratio: {compression_ratio:.1f}x smaller')
    
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    
    optimizer_student = optim.Adam(student.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_student, 'max', patience=3, factor=0.5, verbose=True)
    
    best_acc = 0.0
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    
    print(f"\nStarting knowledge distillation training...")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_kd(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            optimizer=optimizer_student,
            ce_criterion=ce_criterion,
            kl_criterion=kl_criterion,
            temperature=temperature,
            alpha=alpha,
            epoch=epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        test_loss, test_acc = test(student, test_loader, ce_criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), 'student_best.pth')
            print(f"✓ New best student model saved! Accuracy: {best_acc:.2f}%")
        
        print(f'Epoch: {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | ' \
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%')
    
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    
    teacher_test_loss, teacher_test_acc = test(teacher, test_loader, ce_criterion)
    
    print(f'✓ Student training completed!')
    print(f'✓ Teacher accuracy: {teacher_test_acc:.2f}%')
    print(f'✓ Student accuracy: {best_acc:.2f}%')
    print(f'✓ Performance retention: {(best_acc/teacher_test_acc)*100:.1f}%')
    print(f'✓ Model compression: {compression_ratio:.1f}x smaller')
    print(f'✓ Student model saved as: student_best.pth')
    print("=" * 50)
    
    plot_training_history(train_losses, train_accs, test_losses, test_accs, 'student_training_history.png')
    print("✓ Training history plot saved as: student_training_history.png")


if __name__ == '__main__':
    main()
