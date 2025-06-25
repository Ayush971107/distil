# Train larger model on CIFAR10 (Around 15M parameters)
# Train a smaller model (without distil) [$437,034$ parameters]
# Train same arch model with distil

loss = weighed sum of soft targets + labels


τ -> Dividing logits by τ smooths (flattens) the teacher’s output distribution.
This makes the student learn more from the distribution of the teacher (also learning shared traits between the classes)
