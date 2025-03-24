# Denoising Diffusion Probabilistic Models (DDPM) Cheatsheet

DDPMs are generative models that learn to transform random noise into data samples (e.g., images) by modeling a diffusion process. The framework consists of two main processes: the **forward diffusion process**, which adds noise to data, and the **reverse diffusion process**, which learns to remove that noise. Let’s break this down step-by-step.

---

## 1. Forward Diffusion Process

### Purpose
The forward diffusion process takes an original data sample $x_0$ (e.g., an image) and progressively adds Gaussian noise over $T$ time steps, creating a sequence $x_1, x_2, \dots, x_T$. By the end, $x_T$ is nearly indistinguishable from pure noise. This process defines a Markov chain where each step depends only on the previous one, setting up a tractable way to corrupt data that the model will later learn to reverse.

### Step-wise Formula
$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$
- **Explanation**: At each step $t$, the process generates $x_t$ by taking the previous step’s data $x_{t-1}$, scaling its magnitude by $\sqrt{1 - \beta_t}$ (a value slightly less than 1), and adding Gaussian noise with variance $\beta_t$. The scaling ensures that the data doesn’t grow uncontrollably, while the noise gradually corrupts it. This step-by-step addition of noise is what makes the process Markovian.

### Direct Formula
$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$
- **Explanation**: This formula lets us jump directly from the original data $x_0$ to any step $t$ without computing intermediate steps. It shows that $x_t$ is a blend of the original data (scaled by $\sqrt{\bar{\alpha}_t}$) and Gaussian noise (with variance $1 - \bar{\alpha}_t$). This is computationally efficient and crucial for training, as it allows sampling noisy data at any $t$ directly.

### Key Parameters
- **$\beta_t$**: The variance schedule at step $t$.  
  - **Role**: Determines how much noise is added at each step. In DDPM, $\beta_t$ is predefined and typically increases over time (e.g., linearly from $10^{-4}$ to $0.02$ over 1000 steps). Early steps add little noise, preserving most of the data, while later steps add more, pushing $x_t$ toward pure noise.
  - **Context in Learning**: The schedule shapes how quickly the data becomes noise. A well-designed $\beta_t$ ensures the reverse process has enough steps to learn meaningful denoising, balancing training efficiency and sample quality.

- **$\alpha_t = 1 - \beta_t$**: The signal retention factor at step $t$.  
  - **Role**: Represents the fraction of the previous step’s signal kept after adding noise. For example, if $\beta_t = 0.01$, then $\alpha_t = 0.99$, meaning 99% of $x_{t-1}$’s amplitude is retained.
  - **Context in Learning**: $\alpha_t$ controls the gradual degradation of the signal. The model relies on this to understand how much of the original data remains at each step, aiding in the reverse process’s ability to recover it.

- **$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$**: The total signal retention after $t$ steps.  
  - **Role**: This is the product of all $\alpha_s$ values from step 1 to $t$. It quantifies how much of the original data $x_0$ remains in $x_t$. For instance, if $\alpha_s = 0.99$ each step, after 100 steps, $\bar{\alpha}_{100} \approx 0.366$, meaning only 36.6% of the original signal’s amplitude survives, with noise dominating.
  - **Context in Learning**: In the forward process, $\bar{\alpha}_t$ defines the mixture of data and noise at step $t$. During training, the model uses $\bar{\alpha}_t$ to compute $x_t$ from $x_0$, and in the reverse process, it helps scale the noise prediction to match the corruption level. As $t$ increases and $\bar{\alpha}_t$ shrinks, the model learns to denoise from increasingly noisy states, which is key to generating realistic samples from pure noise.

---

## 2. Reverse Diffusion Process

### Purpose
The reverse process starts with pure noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoises it over $T$ steps to approximate the original data distribution $q(x_0)$. A neural network learns to estimate each step, making this the generative part of DDPM.

### Formula
$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \beta_t I)$
- **Explanation**: At each step $t$, the reverse process predicts $x_{t-1}$ as a Gaussian with mean $\mu_\theta(x_t, t)$ (learned by the neural network) and variance $\beta_t$ (fixed to match the forward process). This mirrors the forward process but in reverse, aiming to subtract noise step-by-step.

### Mean Parameterization (ε-Prediction Parameterization)
$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$
- **Explanation**: The neural network predicts the noise $\epsilon_\theta(x_t, t)$ added to create $x_t$. This formula uses that prediction to estimate the mean of $x_{t-1}$, effectively “undoing” the noise addition by adjusting $x_t$ based on the learned noise and the forward process parameters. This is called the ε-prediction parameterization because the network outputs $\epsilon_\theta$ rather than $\mu_\theta$ directly.

### Key Parameters
- **$\epsilon_\theta(x_t, t)$**: The predicted noise at step $t$.  
  - **Role**: The neural network’s output, estimating the noise component in $x_t$. It’s trained to match the actual noise added during the forward process.
  - **Context in Learning**: By predicting $\epsilon_\theta$, the model learns to reverse the diffusion. Accurate noise prediction allows the reverse step to subtract noise correctly, reconstructing the data. This is the core of DDPM’s training objective and the ε-prediction approach.

- **$\mu_\theta(x_t, t)$**: The learned mean of the reverse step.  
  - **Role**: Combines the current noisy sample $x_t$ with the predicted noise $\epsilon_\theta$ to estimate $x_{t-1}$. The formula scales terms using $\alpha_t$ and $\bar{\alpha}_t$ to align with the forward process.
  - **Context in Learning**: The parameterization ensures the reverse process mirrors the forward process’s noise schedule, enabling step-wise denoising. The model optimizes $\theta$ to make $\mu_\theta$ accurate, driving the generation process.

---

## 3. Training

### Objective
Train the neural network to predict the noise $\epsilon$ added to $x_0$ to produce $x_t$ at any step $t$. This allows the reverse process to denoise effectively using the ε-prediction parameterization.

### Loss Function
$\mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$
- **Explanation**: This mean-squared error compares the true noise $\epsilon$ (sampled during the forward process) to the network’s prediction $\epsilon_\theta(x_t, t)$. Minimizing this loss teaches the model to denoise across all $t$, aligning with the ε-prediction strategy.

### Training Steps
1. **Sample $t$** uniformly from $1$ to $T$.
2. **Sample $x_0$** from the real data distribution $q(x_0)$.
3. **Sample noise** $\epsilon \sim \mathcal{N}(0, I)$.
4. **Compute $x_t$**: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
5. **Optimize** $\theta$ to minimize the loss between $\epsilon$ and $\epsilon_\theta(x_t, t)$.

- **Context**: This process simulates the forward diffusion and trains the model to reverse it. $\bar{\alpha}_t$ determines the noise level at each $t$, and the loss ensures the model learns the noise structure across the entire process via ε-prediction.

---

## 4. Sampling

### Purpose
Generate new samples by starting from noise and applying the learned reverse process.

### Procedure
1. **Initialize** $x_T \sim \mathcal{N}(0, I)$.
2. **For $t = T$ down to 1**:
   - Compute $\epsilon_\theta(x_t, t)$.
   - Update:  
     $x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sqrt{\beta_t} z$
     where $z \sim \mathcal{N}(0, I)$ adds stochasticity.
3. **Output** $x_0$ as the generated sample.

- **Explanation**: This iteratively refines noise into data. The $z$ term ensures the variance matches the forward process, maintaining the probabilistic nature of the model.

---

## Parameter Summary and Contextual Roles
- **$\beta_t$**: Dictates the noise added per step. Its schedule (e.g., increasing over time) ensures $x_T$ is noise, giving the model a clear starting point for generation.
- **$\alpha_t = 1 - \beta_t$**: Tracks signal retention per step, linking the forward and reverse processes mathematically.
- **$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$**: Measures the total signal left after $t$ steps. It’s critical for scaling $x_t$ in training and denoising in sampling, telling the model how noisy the data is and how much noise to remove.
- **$\epsilon_\theta(x_t, t)$**: The model’s guess at the noise, enabling the reverse process to peel back layers of corruption via the ε-prediction parameterization.
- **$\mu_\theta(x_t, t)$**: The predicted mean, orchestrating each denoising step using the noise prediction and forward process parameters.

---

## Intuition
- **Forward Process**: Like fading a photo into static—$\bar{\alpha}_t$ tracks how much of the photo remains.
- **Reverse Process**: Like restoring the photo by guessing and removing the static, guided by $\epsilon_\theta$.
- **Learning**: The model masters this restoration by practicing on noisy versions of real data, using $\beta_t$ and $\bar{\alpha}_t$ to navigate the noise levels, with ε-prediction simplifying the task.

This cheatsheet provides a thorough understanding of DDPMs, explaining each parameter’s role and significance in the model’s operation and learning process. It’s a solid foundation for studying the original paper!
