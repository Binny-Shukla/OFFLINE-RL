# **Conservative Q-Learning (CQL) – Offline RL**

This repository contains a self-contained Jupyter Notebook implementing Conservative Q-Learning (CQL) for offline reinforcement learning.
It is designed for quick experimentation and understanding of the CQL algorithm without requiring D4RL or complex environment setups.

## **Overview**

CQL is an offline RL algorithm that:

    Penalizes the Q-function to avoid overestimation on unseen actions.
    
    Learns a conservative critic and actor from a fixed dataset.
    
    Works without online environment rollouts, purely from HDF5 datasets.
    

This notebook includes:

    Full training loop (policy and critic updates).
    
    TD3-style policy delay and target smoothing.
    
    TensorBoard logging and Matplotlib visualizations for losses.
    
    Final plots for Policy Loss and Critic (Q) Loss.

##### How to Use

Open the notebook:

    bash
    jupyter notebook CQL_OfflineRL.ipynb
   
   Run all cells:
   
    The dataset is loaded from HDF5.
    
    Training runs for the specified number of epochs (default: epochs=30).
    
    Loss curves are automatically logged to TensorBoard and saved as PNGs.

#### **Visualize results:**

    bash
    tensorboard --logdir ./runs/CQL

Results
By epoch 30:

Q Loss stabilizes near 1.0 after an initial peak.

Policy Loss converges around -100, indicating stable learning.

**Policy loss :**


<img width="989" height="590" alt="33d78102-f7d6-46ce-b3c2-3b1c4f7d4182" src="https://github.com/user-attachments/assets/cc1dca97-d371-4c3a-8999-8dc9f4f773e1" />


** Q Loss:**

<img width="989" height="590" alt="a48746aa-9884-4ec2-b18a-88212d2f9b3e" src="https://github.com/user-attachments/assets/ea5296af-7cba-4bd4-96d6-dcb7fd4fcd7d" />
