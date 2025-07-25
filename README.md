# Knowledge Distillation on CIFAR-10

This repository implements knowledge distillation for training smaller student models using larger teacher models on CIFAR-10.

## Repository Structure

- **`base_template.py`**: Contains model architectures (Teacher & Student) and common training utilities
- **`train_teacher.py`**: Script for training the larger teacher model  
- **`train_student.py`**: Script for training the smaller student model using knowledge distillation

## Usage

### 1. Train Teacher Model
```bash
python train_teacher.py
```
This trains a larger CNN (teacher) and saves the best model as `teacher_best.pth`.

### 2. Train Student Model (Knowledge Distillation)
```bash
python train_student.py
```
This trains a smaller CNN (student) using knowledge distillation from the pre-trained teacher.

## Knowledge Distillation Details

**Loss Function**: `loss = α × KD_loss + (1-α) × CE_loss`
- KD_loss: KL divergence between softened teacher and student outputs
- CE_loss: Standard cross-entropy with ground truth labels
- α: Weight balancing soft vs hard targets

**Temperature (τ)**: Divides logits to smooth probability distributions, helping the student learn from the teacher's uncertainty.
