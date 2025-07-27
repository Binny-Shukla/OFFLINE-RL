# **Offline RL Collection**

This repository is a research-focused collection of Offline Reinforcement Learning (RL) algorithms, implemented from scratch in PyTorch with a focus on clarity, technical depth, and reproducibility.

Each algorithm is maintained on its own branch for clean isolation and easy benchmarking. This setup allows us to evolve and experiment with each approach independently while maintaining a unified structure.

**Implemented Algorithms**

## **Behavior Cloning (BC)**
A supervised learning baseline for offline RL, trained directly on dataset actions without environment interaction.
Branch: bc

## **Implicit Q-Learning (IQL)**

An advantage-weighted actor-critic algorithm designed for high-quality offline RL without explicit behavior regularization.
Branch: iql

### Why This Repo?

Most open-source implementations either:

Abstract away the math behind wrappers, or

Mix multiple algorithms together, making it hard to study them in isolation.

 #### Here, each branch:

Implements the algorithm from scratch using clean PyTorch.

Includes mathematical derivations (as comments and in the README).

Uses TensorBoard logging and easily exportable results for papers.

#### How to Use

Each branch has:

A Jupyter Notebook (*_training.ipynb) with the full training pipeline.

TensorBoard support for monitoring.

A README explaining the loss functions, math, and performance.

To clone a specific algorithm branch:

git clone -b <branch-name> https://github.com/Binny-Shukla/offline-rl-collection.git

cd offline-rl-collection
