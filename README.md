# **🎯 Decision Diffuser (DD) – Offline Reinforcement Learning**

This repository contains a self-contained Jupyter Notebook implementing Decision Diffusers (DD) for offline reinforcement learning. It’s built from scratch for experimentation and demonstrates how generative diffusion models can be applied to model RL trajectories.

## **🔎 Overview**

Decision Diffuser reframes RL as a conditional generative modeling problem over trajectories:

Uses diffusion models to learn the distribution of optimal trajectories.

Generates actions by denoising sampled trajectories, conditioned on goals or rewards.

Operates fully offline – trained on pre-collected datasets, no environment interaction required.

This notebook includes:

✅ Complete scratch implementation of Decision Diffuser.

✅ Custom dataset builder for trajectory conditioning.

✅ Diffusion training loop with noise scheduling.

✅ Matplotlib utilities for clean training & denoising visualizations.

### **🛠 How to Use**

Open the notebook:

    jupyter notebook DecisionDiffuser.ipynb


    Run all cells:

    Loads offline dataset.

Builds trajectories for diffusion training.

Trains the DD model with configurable hyperparameters.

Visualize training:

    python plot.py


Generates smooth loss curves & denoising plots.

### **📊 Results**

By the end of training:

The diffusion loss converges, showing the model learns to denoise trajectories effectively.

Generated trajectories reflect goal-conditioned behavior.

Example loss curve:

🟦 Diffusion Training Loss:

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/35c6d928-ecdd-424b-94d4-086e7e5cfdd6" />


#### **📂 Repo Structure**

│── DecisionDiffuser.ipynb   # Full implementation  

│── plot.py                  # Visualization script  

│── readme.md  

#### **🚀 Key Takeaways**

Decision Diffuser ≠ Q-learning or Transformers.

It’s denoising-based generative modeling for RL.

Offline RL with DD feels like training an image diffusion model, but on state-action trajectories instead of pixels.

#### **🤝 Acknowledgements**

Inspired by the original Decision Diffuser paper (Janner et al., 2022).
