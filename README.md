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

Implemented fully from scratch for clarity & learning.
