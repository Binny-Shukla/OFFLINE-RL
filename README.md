# **🎯 Trajectory Transformer (TT) – Offline Reinforcement Learning**

This repository contains a self-contained Jupyter Notebook implementing Trajectory Transformers (TT) for offline reinforcement learning. It’s built from scratch for quick experimentation and to show how sequence models (like GPT) can be applied to model entire RL trajectories.

## **🔎 Overview**

    Trajectory Transformer reframes RL as a sequence modeling problem over trajectories:
    
    Treats RL data as tokenized sequences: states, actions, rewards.
    
    Predicts future actions & rewards conditioned on past trajectory.
    
    No explicit value functions or policy gradient training – just supervised sequence modeling.
    
    Fully offline: trained on pre-collected datasets, no env rollouts required.

This notebook includes:

✅ Complete scratch implementation of TT (no external libraries like D4RL required).

✅ Custom dataset class for building trajectories from offline data.

✅ Training loop with supervised loss.

✅ Matplotlib utilities for plotting smooth training curves.

### **🛠 How to Use**


Open the notebook:

    jupyter notebook TrajectoryTransformer.ipynb


Run all cells:

Loads offline dataset.
    
    Builds trajectory sequences.
    
    Trains TT with configurable hyperparameters.

Visualize training:

    python plot.py


Generates smooth loss curves with Matplotlib.

### **📊 Results**

By the end of training:

    The TT loss stabilizes, showing convergence of trajectory sequence modeling.
    
    Loss curves are smooth, indicating reliable offline training.

Example loss curve:

**🟦 TT Loss:**

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/43414c51-5bce-4530-bf9a-369510d6abb7" />


### **📂 Repo Structure**

│── TrajectoryTransformer.ipynb   # Full implementation  

│── plot.py                       # Visualization script  

│── readme.md


### **🚀 Key Takeaways**

TT is not Q-learning or actor-critic.

It’s pure supervised learning on trajectory sequences.

Offline RL feels just like training a language model – but on states, actions, and rewards instead of words.

#### **🤝 Acknowledgements**

Inspired by the original Trajectory Transformer paper (Chen et al., 2021).
