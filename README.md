<div align="center">
  <img src="docs/logo.png" alt="Logo: a visualisation of the forward diffusion process" width="70%">

  <p><i>Logo generated with a DDPM-based diffusion model</i></p>

  <h1>Denoising Diffusion Probabilistic Models</h1>
  <h3>A Self-Contained Mathematical Treatment</h3>

  <p>
    <b>Bachelor's Thesis</b> &nbsp;·&nbsp; University of Cassino and Southern Lazio<br>
    Department of Electrical and Information Engineering “Maurizio Scarano”<br>
    B.Sc. in Computer and Telecommunications Engineering &nbsp;·&nbsp; A.Y. 2022/2023
  </p>

  <p>
    <a href="main.pdf"><img src="https://img.shields.io/badge/Thesis-main.pdf-success.svg" alt="Read the thesis"></a>
    <img src="https://img.shields.io/badge/Language-Italian-lightgrey.svg" alt="Written in Italian">
    <img src="https://img.shields.io/badge/Type-Theoretical%20analysis-0052cc.svg" alt="Theoretical analysis">
    <img src="https://img.shields.io/badge/Built%20with-LaTeX-013243.svg" alt="Built with LaTeX">
    <a href="https://arxiv.org/abs/2006.11239"><img src="https://img.shields.io/badge/Primary%20ref-arXiv%3A2006.11239-B31B1B.svg" alt="arXiv:2006.11239"></a>
  </p>
</div>

---

## Overview

This repository contains the LaTeX sources and the compiled PDF of my Bachelor's thesis, *Generatori di immagini da prompt testuali: stato dell'arte* (**Text-to-Image Generators: State of the Art**), supervised by Prof. Alessandro Bria.

The thesis studies **Denoising Diffusion Probabilistic Models** (DDPMs), following Ho, Jain and Abbeel (2020): how gradual corruption can be reversed to generate images, and how this construction leads to a trainable objective. Appendices supply the derivation and mathematical background. Stable Diffusion, Imagen and DALL·E 2 provide the text-to-image context; the framework below describes unconditional DDPMs.

> [!NOTE]
> **The thesis is written in Italian.** Read [`main.pdf`](main.pdf) for the full dissertation. This English overview presents the framework developed in Chapter 2 and Appendix B.

> [!IMPORTANT]
> **Scope.** This is an expository, theoretical study. No new architecture is proposed or model trained; empirical results are attributed to the literature.

---

## What the thesis establishes

**Positioning within generative modelling.** Chapter 1 distinguishes generative from discriminative modelling and classifies models by their representation of $`p_{\boldsymbol{\theta}}(\mathbf{x})`$, following Goodfellow (2016) and Foster (2023). DDPMs define an explicit latent-variable model with a generally intractable marginal likelihood, addressed through variational inference.

**Formalisation of the diffusion mechanism.** Chapter 2 defines the forward and reverse chains, their Gaussian kernels and variance schedule. The marginal $`q(\mathbf{x}_t \mid \mathbf{x}_0)`$ enables direct noising; the posterior $`q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)`$ supports the variational objective. Training and ancestral sampling complete the account.

**Derivation of the training objective.** Appendix B applies the VAE framework of Kingma and Welling to derive and decompose the variational bound, then develops the simplified noise-prediction loss. The final reweighting is a modelling choice, not an algebraic identity.

**Mathematical prerequisites.** Appendix A covers probability distributions, expectation, variance, independence, Gaussians, Markov chains and maximum-likelihood estimation. Appendix C introduces the U-Net of Ronneberger et al. (2015), including downsampling, upsampling and skip connections.

**Ethical discussion.** Chapter 3 examines dataset bias, prompt modification in DALL·E, and synthetic depictions of real events through the Amnesty International and Adobe Stock cases.

---

## The mathematical framework

Let $`\mathbf{x}_0 \sim q(\mathbf{x}_0)`$ be a data sample, $`\mathbf{x}_{1:T}`$ its progressively corrupted latent states, and $`\{\beta_t\}_{t=1}^{T} \subset (0,1)`$ a fixed *variance schedule*. The schedule used here is increasing, although the identities below do not require monotonicity. Define

```math
\alpha_t := 1 - \beta_t, \qquad \bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s .
```

### The forward process: Markovian corruption

The forward process $`q`$ is a **fixed**, non-learned Markov chain that adds noise to $`\mathbf{x}_0`$ over $`T`$ timesteps:

```math
q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) := \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1}),
```

and each transition combines a scaled input with isotropic Gaussian noise:

```math
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) := \mathcal{N}\!\left(\mathbf{x}_t ; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I}\right).
```

The scaling $`\sqrt{1-\beta_t}`$ preserves the standard Gaussian distribution and keeps covariance bounded when the initial covariance is finite.

**Closed-form marginal.** Composing the linear Gaussian transitions allows $`\mathbf{x}_t`$ to be sampled *directly* from $`\mathbf{x}_0`$:

```math
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\ (1-\bar{\alpha}_t)\mathbf{I}\right),
```

equivalently, via the reparameterisation trick,

```math
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
```

Training samples a timestep $`t`$ uniformly and constructs its noisy image directly. Precomputed schedule coefficients eliminate the preceding transitions, leaving image operations and a network evaluation.

**Limiting behaviour.** Because $`\alpha_s \in (0,1)`$ for every $`s`$, $`\bar{\alpha}_t`$ decreases strictly. If $`\bar{\alpha}_T \to 0`$ as the chain length increases, $`q(\mathbf{x}_T \mid \mathbf{x}_0)`$ converges to $`\mathcal{N}(\mathbf{0}, \mathbf{I})`$ for each fixed $`\mathbf{x}_0`$. At finite length, this Gaussian is an approximation with a small residual signal.

### The reverse process: learned Gaussian denoising

The reverse conditional $`q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)`$ depends on the unknown $`q(\mathbf{x}_0)`$ and is generally intractable. Generation therefore uses a **learned** Gaussian Markov chain $`p_{\boldsymbol{\theta}}`$:

```math
\begin{aligned}
p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T}) &:= p(\mathbf{x}_T) \prod_{t=1}^{T} p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t), \qquad p(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \mathbf{I}), \\[4pt]
p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t) &:= \mathcal{N}\!\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_t, t),\ \boldsymbol{\Sigma}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)\right).
\end{aligned}
```

Under suitable regularity conditions, small $`\beta_t`$ justify a local Gaussian approximation. Ho et al. fix $`\boldsymbol{\Sigma}_{\boldsymbol{\theta}}(\mathbf{x}_t,t) = \sigma_t^2 \mathbf{I}`$ with prescribed $`\sigma_t^2`$, learning only the mean. Sampling starts from the Gaussian prior and applies the reverse transitions successively.

**The tractable posterior.** Although $`q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)`$ is generally intractable, conditioning on the training sample $`\mathbf{x}_0`$ gives a closed-form Gaussian for timesteps greater than one:

```math
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_{t-1} ; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\ \tilde{\beta}_t \mathbf{I}\right),
```

where

```math
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t, \qquad \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t .
```

This posterior provides Gaussian targets whose KL divergences are analytic. Expectations over data, noise and timesteps are still estimated stochastically.

### The training objective: from the ELBO to the simplified loss

The likelihood $`p_{\boldsymbol{\theta}}(\mathbf{x}_0) = \int p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T})\,\mathrm{d}\mathbf{x}_{1:T}`$ is intractable. Treating $`\mathbf{x}_0`$ as observed and $`\mathbf{x}_{1:T}`$ as latent gives the variational upper bound on its negative logarithm:

```math
-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0) \ \leq\ \mathbb{E}_{q}\!\left[-\log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\right] \ =:\ L_{\mathrm{vlb}} .
```

Appendix B decomposes $`L_{\mathrm{vlb}}`$ into prior matching, intermediate Gaussian KL divergences, and a reconstruction negative log-likelihood. Below, expectation over the forward trajectory conditional on the observed sample is implicit on the right; the sum ends at the terminal timestep:

```math
L_{\mathrm{vlb}} = \underbrace{D_{\mathrm{KL}}\!\left(q(\mathbf{x}_T\mid\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\right)}_{L_T} + \sum_{t>1} \underbrace{D_{\mathrm{KL}}\!\left(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\right)}_{L_{t-1}} \underbrace{-\ \log p_{\boldsymbol{\theta}}(\mathbf{x}_0\mid\mathbf{x}_1)}_{L_0} .
```

The parameter-independent $`L_T`$ can be omitted during optimisation. With fixed variances, each $`L_{t-1}`$ is a weighted squared error between $`\tilde{\boldsymbol{\mu}}_t`$ and $`\boldsymbol{\mu}_{\boldsymbol{\theta}}`$, up to parameter-independent terms. Expressing $`\boldsymbol{\mu}_{\boldsymbol{\theta}}`$ through predicted noise $`\boldsymbol{\epsilon}`$ and dropping timestep weights gives the simplified criterion, applied across all timesteps:

```math
L_{\mathrm{simple}}(\boldsymbol{\theta}) := \mathbb{E}_{\mathbf{x}_0,\ \boldsymbol{\epsilon},\ t}\left[\left\lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\!\left(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\ t\right)\right\rVert^2\right],
```

with $`\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})`$ and $`t \sim \mathcal{U}\{1,\dots,T\}`$ sampled independently of each other and the data. The expectation includes the data distribution; $`L_{\mathrm{simple}}`$ is a surrogate, not the original variational bound.

The network learns to predict the noise added to a training image, without manual labels. Its predictions determine the reverse transition means. Ho et al. connect this objective to denoising score matching.

---

## Repository layout

| Path | Contents |
| :--- | :--- |
| [`main.pdf`](main.pdf) | **Compiled thesis.** Start here. |
| `main.tex` | Master document: preamble, package configuration, inclusion order. |
| `customization.sty` | Custom preamble: theorem environments, styling, macros. |
| `FrontMatter/` | Title page, dedication, abstract. |
| `MainMatter/Introduzione/` | Introduction. |
| `MainMatter/Chapters/Chap1/` | Ch. 1, *Generative modelling*: generative vs. discriminative, taxonomy of generative models. |
| `MainMatter/Chapters/Chap2/` | Ch. 2, *Diffusion models*: the DDPM pipeline, forward and reverse processes, training, sampling, reported performance. |
| `MainMatter/Chapters/Chap3/` | Ch. 3, *Conclusions*, including the ethical discussion. |
| `MainMatter/Appendix/` | App. A, probability and statistics; App. B, derivation of the DDPM loss; App. C, the U-Net. |
| `Images/` | Figures (PDF/SVG) and standalone TikZ sources. |
| `BackMatter/` | Bibliography (`Tesi.bib`) and acknowledgements. |
| `docs/` | Assets for this README. |

---

## Building from source

Install a TeX distribution with the packages in `main.tex` and `customization.sty`, **biber**, and **Inkscape** on the `PATH`. From the repository root, run the build encoded by the [`arara`](https://islandoftex.github.io/arara/) directives:

```bash
arara main.tex
```

To run the same sequence manually:

```bash
pdflatex -shell-escape main.tex
biber main
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
```

Shell escape enables SVG conversion through Inkscape. The final LaTeX passes resolve references and bibliography; repeat if requested by the log.

---

## References

The thesis builds on the following works. The complete bibliography is in `BackMatter/Tesi.bib`.

- J. Ho, A. Jain, P. Abbeel. **Denoising Diffusion Probabilistic Models.** *NeurIPS*, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239). The primary reference for the DDPM formulation and training objective.
- J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli. **Deep Unsupervised Learning using Nonequilibrium Thermodynamics.** *ICML*, 2015. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585). The earlier diffusion probabilistic framework, motivated by nonequilibrium thermodynamics.
- P. Dhariwal, A. Nichol. **Diffusion Models Beat GANs on Image Synthesis.** *NeurIPS*, 2021. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233). A study of diffusion-model image quality relative to GAN baselines.
- D. P. Kingma, M. Welling. **Auto-Encoding Variational Bayes.** *ICLR*, 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114). The variational bound specialised in Appendix B.
- O. Ronneberger, P. Fischer, T. Brox. **U-Net: Convolutional Networks for Biomedical Image Segmentation.** *MICCAI*, 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597). The architecture of Appendix C.
- I. Goodfellow. **NIPS 2016 Tutorial: Generative Adversarial Networks.** [arXiv:1701.00160](https://arxiv.org/abs/1701.00160). The taxonomy adopted in Chapter 1.
- D. Foster. **Generative Deep Learning.** 2nd ed., O'Reilly, 2023.

---

## Citation

```bibtex
@thesis{alfieri2023ddpm,
  author      = {Alfieri, Giuseppe},
  title       = {Generatori di immagini da prompt testuali: stato dell'arte},
  type        = {Bachelor's Thesis},
  institution = {Universit\`a degli Studi di Cassino e del Lazio Meridionale},
  address     = {Cassino, Italy},
  year        = {2023},
  url         = {https://github.com/giusalfieri/Bachelor-Thesis}
}
```

---

<div align="center">
  <sub><b>Giuseppe Alfieri</b> &nbsp;·&nbsp; Supervisor: Prof. Alessandro Bria &nbsp;·&nbsp; Academic Year 2022/2023</sub>
</div>
