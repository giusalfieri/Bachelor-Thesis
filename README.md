<div align="center">
  <img src="docs/logo.png" alt="DDPM Logo - Forward Diffusion Process Visualized" width="100%">

  <h1>Denoising Diffusion Probabilistic Models (DDPM)</h1>
  <h3>A Rigorous Theoretical Analysis & State of the Art Review</h3>

  <p>
    <b>Bachelor's Thesis Project</b> | University of Cassino and Southern Lazio
    <br />
    Computer and Telecommunications Engineering
  </p>

  <img src="https://img.shields.io/badge/Focus-Theoretical%20Analysis-0052cc.svg" alt="Theoretical Focus">
  <img src="https://img.shields.io/badge/Math-Stochastic%20Processes-013243.svg" alt="Math">
  <a href="https://arxiv.org/abs/2006.11239">
    <img src="https://img.shields.io/badge/Ref-arXiv%3A2006.11239-B31B1B.svg" alt="arXiv Reference">
  </a>
</div>

<br />

---

## 📖 Abstract

This repository hosts the theoretical framework and research conducted for my Bachelor's thesis, titled **"Text-to-Image Generators: State of the Art"**.

This work is a comprehensive theoretical dissertation that moves beyond black-box applications to provide a rigorous mathematical deconstruction of **Denoising Diffusion Probabilistic Models (DDPMs)**. It explores the intersection of non-equilibrium thermodynamics, stochastic processes, and variational inference that forms the bedrock of modern generative AI.

### Key Contributions
* **State of the Art Review:** A technical analysis of major architectures including DALL-E 2, Stable Diffusion, and Imagen (Chapter 2).
* **Mathematical Foundations:** A deep dive into DDPMs as coupled Markov chains, detailing the probabilistic transition kernels involved in data corruption and restoration (Chapter 3).
* **Variational Derivation:** Step-by-step derivation of the Evidence Lower Bound (ELBO) to justify the simplified training objective used in practice (Appendix B).

---

## 🧠 Theoretical Framework Deconstructed

The core of the thesis explores how diffusion models learn the data distribution $p(x_0)$ by reversing a gradual, structured noising process.

<details>
<summary><b>▶️ The Forward Diffusion Process (Markovian Corruption)</b> (Click to expand)</summary>
<br />

The forward process, denoted as $q$, is defined as a fixed Markov chain that gradually adds Gaussian noise to the data $x_0$ over $T$ timesteps, according to a predefined variance schedule $\beta_t$.

The joint probability of the complete sequence is factorized as:

$$q(x_{1:T} | x_0) := \prod_{t=1}^T q(x_t | x_{t-1})$$

Where each individual transition is a Gaussian kernel ensures mathematical tractability:

$$q(x_t | x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$$

A critical property derived in the work is the ability to sample $x_t$ at any arbitrary timestep directly from $x_0$ via the **reparameterization trick**, without iterating through intermediate steps:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

As $T \to \infty$, the data structure is completely destroyed, converging to pure isotropic Gaussian noise.
</details>

<details>
<summary><b>◀️ The Reverse Process & Optimization Objective</b> (Click to expand)</summary>
<br />

Generative modeling requires inverting the diffusion. Since the true posterior $q(x_{t-1}|x_t)$ is intractable, we approximate it using a learned Markov chain $p_\theta$ with parameterized Gaussian transitions:

$$p_\theta(x_{t-1} | x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Training is performed by minimizing the variational upper bound on the negative log-likelihood. The thesis details the complex derivation showing that this objective can be simplified to learning a neural network $\epsilon_\theta$ that predicts the noise component $\epsilon$ added at timestep $t$.

This results in the highly effective **simple loss function**:

$$
L_{\text{simple}}(\theta) \coloneqq
\mathbb{E}_{x_0,\epsilon,t}
\left[
\left\lVert
\epsilon -
\epsilon_\theta\!\left(
\sqrt{\bar{\alpha}_t}\,x_0 +
\sqrt{1-\bar{\alpha}_t}\,\epsilon,\,
t
\right)
\right\rVert^{2}
\right]
$$


This formulation elegantly reduces the complex generative task to a sequence of denoising score matching problems.
</details>

---

## 📂 Thesis Structure

The associated dissertation is structured as follows:
* **Chapter 1:** Introduction to Generative Deep Learning context.
* **Chapter 2:** Technical review of SOTA Text-to-Image architectures.
* **Chapter 3:** Mathematical formulation of DDPMs (Forward/Reverse processes).
* **Appendices:** Deep dives into VAEs, ELBO derivation, and U-Net architecture.

## 📚 Foundational Reference

The theoretical backbone of this thesis relies heavily on the seminal paper that established modern DDPMs:

> **Denoising Diffusion Probabilistic Models**<br>
> Jonathan Ho, Ajay Jain, Pieter Abbeel<br>
> *NeurIPS 2020*<br>
> [**➡️ Read on arXiv (2006.11239)**](https://arxiv.org/abs/2006.11239)

---
<div align="center">
    <p><i>Author: [Giuseppe Alfieri] | Supervisor: Prof. Alessandro Bria</i></p>
    <p>Academic Year 2022/2023</p>
</div>
