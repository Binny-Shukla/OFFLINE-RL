
# **TD3 + BC (Offline Reinforcement Learning)**

This branch contains an implementation of TD3 + BC (Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning), a state-of-the-art algorithm for offline reinforcement learning. The model is trained entirely on a fixed dataset (no environment interaction) using h5py-stored transitions.

### What’s Inside

TD3 + BC Implementation:

Core actor-critic architecture, target networks, and behavior cloning regularization.

#### Training Notebook:

A single Jupyter notebook for ease of experimentation — trains the policy, logs with TensorBoard, and visualizes loss curves.

### **Loss Curves:**

**Policy Loss** *(converging around -120)*

<img width="583" height="455" alt="8ab94b34-2026-4e24-aed5-7695dbaef4e9" src="https://github.com/user-attachments/assets/1b5216a0-80eb-466b-b1b5-33b8a090f555" />


**Critic Loss** *(converging around 1)*

<img width="567" height="455" alt="c4741975-b9db-4e3e-8d9f-c059f8fb18e7" src="https://github.com/user-attachments/assets/9fd52719-aac8-4093-b379-e38f18def12f" />


Both plotted over 30 epochs for stability and reproducibility.

**MIT License:**

Fully open-source and free to use or extend.

### **Why TD3 + BC?**

Traditional TD3 struggles in offline RL because it tends to extrapolate out-of-distribution actions. TD3 + BC stabilizes learning by combining:

Policy optimization (TD3) for exploitation, and

Behavior cloning loss to keep actions close to the dataset distribution.

This balance improves robustness and avoids divergence.

### **How to Use**

Clone the repo and checkout the td3+bc branch:

    bash
    git clone <https://github.com/Binny-Shukla/OFFLINE-RL>
    cd <OFFLINE RL>
    git checkout td3+bc
    
Run training:

    bash
    jupyter notebook TD3_BC_Offline.ipynb

Launch TensorBoard for monitoring:

    bash
    tensorboard --logdir="./runs/TD3_BC"
    
**Training Results** *(30 Epochs)*

Policy Loss steadily decreased to ~-120, indicating convergence.

Critic Loss dropped to ~0.8–1.0, showing stable Q-function learning.


**Notes**

The dataset is static; no environment interaction is required.

For efficiency, training is capped at 30 epochs (50+ would overfit given our dataset size).

Model weights can be saved using torch.save() after training.

=======
# **🎯 Decision Transformer (DT) – Offline Reinforcement Learning**

This repository contains a self-contained Jupyter Notebook implementing Decision Transformers (DT) for offline reinforcement learning.
It’s built from scratch, designed for quick experimentation and understanding of how sequence models (like GPT/BERT) can be applied to RL.

# **🔎 Overview**

Decision Transformer reframes RL as a sequence modeling problem:

Uses Trajectory Transformer idea – predicts future actions from past states, actions, and return-to-go.

No explicit value functions or rewards during training – instead, it’s trained like a supervised sequence model.

Can be run fully offline using HDF5 datasets, without environment rollouts.

This notebook includes:

    ✅ Complete scratch implementation of DT (no external libraries like D4RL required).
    
    ✅ HDF5 dataset loader (h5py).
    
    ✅ Training loop with supervised loss (cross-entropy / MSE depending on setting).
    
    ✅ plot.py utility for TensorBoard + Matplotlib graphs.

### **🛠 How to Use**

    Open the notebook:
    
    jupyter notebook DecisionTransformer.ipynb
    
    
    Run all cells:
    
    Loads dataset from .h5 file.
    
    Trains DT with configurable hyperparameters.
    
    Logs training via TensorBoard + plots.

Visualize results:

    tensorboard --logdir ./runs/DT

## **📊 Results**

By the end of training:

The loss stabilizes, indicating convergence of sequence modeling.

Training vs. validation curves look nearly identical (since DT is purely deterministic & offline).

Example loss curve:

## **🟦 Policy/Sequence Loss:**

<img width="640" height="480" alt="log_Figure_1" src="https://github.com/user-attachments/assets/2cf79a9d-06ca-44e5-abcf-f520ce4e3469" />

#### **📂 Repo Structure**

│── DecisionTransformer.ipynb   # full implementation  

│── plot.py                     # visualization script  

│── runs/                       # tensorboard logs  

#### **🚀 Key Takeaways**

    DT is not a Q-learning or actor-critic method.
    
    It’s pure supervised learning over trajectories.
    
    Offline RL datasets → Inference feels the same for training & validation.

#### **🤝 Acknowledgements**

Inspired by the original Decision Transformer paper (Chen et al., 2021)

Implemented fully from scratch for clarity & learning
