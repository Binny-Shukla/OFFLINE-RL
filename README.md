# **Behavior Cloning (BC)**

This branch contains our implementation of Behavior Cloning (BC), a simple yet foundational offline reinforcement learning (RL) algorithm.
BC serves as a baseline for learning policies directly from fixed datasets without any environment interaction.

## **Overview**

Behavior Cloning treats offline RL as a supervised learning problem, directly imitating the behavior policy that generated the dataset.

The policy (actor network) is trained by minimizing the Mean Squared Error (MSE) between predicted actions and dataset actions:

    python
    loss = MSE(pred_actions, actions)
    
There is no critic, value function, or advantage estimation — making BC fast to train but prone to distributional shift in more complex tasks.

### **Training Details**

    Optimizer: AdamW (no weight decay)
    
    Learning Rate: 3e-4
    
    Batch Size: 256
    
    Epochs: 30
    
    Scheduler: Cosine Annealing (T_max=30)
    
    Gradient Clipping: max_norm=0.5
    
    Loss is logged via TensorBoard (./runs/bc) and visualized for each epoch.

### **Performance**
Performance is measured by training loss across epochs.
Below is the trend of BC loss (MSE) over 30 epochs:

<img width="2400" height="1500" alt="image" src="https://github.com/user-attachments/assets/68cc6cdc-6f2f-49d9-b63b-934fcb419176" />



#### **Usage**


1. **Install dependencies**:

        bash
        pip install torch torchvision tensorboard tqdm
    
2. **Train BC**:

        bash
        python train_bc.ipynb
    
3. **Launch TensorBoard**:

        bash
        tensorboard --logdir=./runs/bc
