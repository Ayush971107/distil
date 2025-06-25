# Train larger model on CIFAR10 (Around 15M parameters)
# Train a smaller model (without distil) [$437,034$ parameters]
# Train same arch model with distil

loss = weighed sum of soft targets + labels


τ -> Dividing logits by τ smooths (flattens) the teacher’s output distribution.
This makes the student learn more from the distribution of the teacher (also learning shared traits between the classes)

Hyper-parameter ablation

Sweep a grid over τ ∈ {2, 4, 8} and α ∈ {0.5, 0.7, 0.9}.

Plot test-accuracy vs. those settings to see how sensitive KD is to your choices.

Student capacity study

Try two smaller students (e.g. halve the channel counts, or remove one Conv block) and one larger student (add extra filters).

Train each with and without KD. You’ll observe how the teacher helps (or doesn’t) as the student capacity changes.

next -> run more experiments (take inspiration from the paper)