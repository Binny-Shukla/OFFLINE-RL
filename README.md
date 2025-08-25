# **Implicit Q-Learning (IQL)**

This branch contains our implementation of Implicit Q-Learning (IQL), an offline reinforcement learning algorithm designed for settings where no environment interaction is allowed during training. IQL leverages value-weighted policy updates to avoid explicit policy constraints, making it a stable and practical baseline for real-world offline RL.

### **Overview**

IQL decomposes the learning process into three main objectives:

1. **Value Function (V) Update**
   
    We learn a value function V(s) by regressing towards a quantile target:

        delta = r + gamma * V(s') - V(s)
        weight = tau if delta > 0 else (1 - tau)
        value_loss = (weight * delta**2).mean()
   
2. **Q-Function (Critic) Update**
 
    We learn two Q-functions Q1, Q2 using standard TD targets:

        target = r + gamma * V(s')
        q_loss = 0.5 * [MSE(Q1(s,a), target) + MSE(Q2(s,a), target)]
   
3. **Policy Update**
 
    Instead of maximizing Q directly, the policy is updated with advantage-weighted behavior cloning:

        actions ~ pi(a|s)
        advantages = min(Q1, Q2) - V(s)
        weights = exp(beta * advantages)  # clipped to 100.0
        policy_loss = -(weights * log_probs(actions|s)).mean()
   
#### **Training Details**

Optimizers: AdamW (no weight decay)

Learning rates:

    Policy: 1e-4
    
    Value/Q: 1e-3

Cosine Annealing LR Scheduler (T_max=100)

##### **Hyper Params**

    Batch size: 256
    
    Discount factor gamma = 0.97
    
    Quantile parameter tau = 0.7
    
    Temperature beta = 2.0
    

## **Performance**
Performance metrics (value loss, Q-loss, policy loss) are logged in TensorBoard.

### Policy Loss:

<img width="784" height="452" alt="image" src="https://github.com/user-attachments/assets/94013712-f4d1-4d3b-b3ed-8ef43a318362" />



### Value Loss:

<img width="751" height="452" alt="image" src="https://github.com/user-attachments/assets/20b56ffc-0353-4eb5-a4dd-4ae57c1d8a63" />


### Q Loss: 

<img width="751" height="452" alt="image" src="https://github.com/user-attachments/assets/fb9876b1-300f-4eee-9054-9135ab38410a" />


