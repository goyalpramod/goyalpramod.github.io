<!-- ---
layout: blog
title: "Demystifying Diffusion Models"
date: 2025-01-3 12:00:00 +0530
categories: [personal, technology]
image: assets/blog_assets/demystifying_diffusion_models/temp_meme_img.webp
---
References:

[Lil'Log blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
[HF article](https://huggingface.co/blog/annotated-diffusion) (code + notes)
[yang song](https://yang-song.net/blog/2021/score/)
[Fast ai course](https://course.fast.ai/Lessons/part2.html)

https://huggingface.co/blog/stable_diffusion

## Forward Diffusion process 

Imagine you have a large dataset of images, we will represent this real data distribution as $q(x)$ and we take an image from it (data point) $x_0$.
(Which is mathematically represented as  $x_0 \sim q(x)$).

In the forward diffusion process we add small amounts of Gaussian noise to the image ($x_0$) in $T$ steps. Which produces a bunch of noisy images as each step which we can label as $x_1,\ldots,x_T$. These steps are controlled by a variance schedule given by $\beta_t$. The value of $\beta_t$ ranges from 0 to 1 (i.e it can take values like 0.002, 0.5,0.283 etc) for $t, \ldots, T$. (Mathematically represented as ${\beta_t \in (0,1)}_{t=1}^T$)

There are many reasons we choose Gaussian noise, but it's mainly due to the properties of normal distribution. (about which you can read more here)

Now let us look at the big scary forward diffusion equation and understand what is going on


$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$$
$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$$

{add quation 1 and 2 label}

$q(x_t|x_{t-1})$ means that given that I know $q(x_{t-1})$ what is the probability of $q(x_t)$ This is also knows as [bayes theorem](). 

To simplify it, think of it as. given $q(x_0)$ (for value of $t$ = 1) I know the value of $q(x_1)$.

The right handside of equation 1 represents a normal distribution. 

(is the lhs a probability or a distribution? if it is a distribution how can we use bayes theorem with it?)

The data sample $x_0$ gradually loses its distinguishable features as the step $t$ becomes larger. Eventually when $T \to \infty$, $x_T$ is equivalent to an isotropic Gaussian distribution.\
"""

"""\
A nice property of the above process is that we can sample $x_t$ at any arbitrary time step $t$ in a closed form using reparameterization trick. Let $\alpha_t = 1-\beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$:

$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}$ ; where $\epsilon_{t-1}, \epsilon_{t-2}, \cdots \sim \mathcal{N}(0,\mathbf{I})$
$= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\bar{\epsilon}_{t-2}$ ; where $\bar{\epsilon}_{t-2}$ merges two Gaussians (*)
$= \cdots = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$

$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$

(*) Recall that when we merge two Gaussians with different variance, $\mathcal{N}(0,\sigma_1^2\mathbf{I})$ and $\mathcal{N}(0,\sigma_2^2\mathbf{I})$, the new distribution is $\mathcal{N}(0,(\sigma_1^2+\sigma_2^2)\mathbf{I})$. Here the merged standard deviation is $\sqrt{(1-\alpha_t)+\alpha_t(1-\alpha_{t-1})} = \sqrt{1-\alpha_t\alpha_{t-1}}$.

Usually, we can afford a larger update step when the sample gets noisier, so $\beta_1 < \beta_2 < \cdots < \beta_T$ and therefore $\bar{\alpha}_1 > \cdots > \bar{\alpha}_T$.\
"""

"""\
**Connection with stochastic gradient Langevin dynamics**\
Langevin dynamics is a concept from physics, developed for statistically modeling molecular systems. Combined with stochastic gradient descent, stochastic gradient Langevin dynamics (Welling & Teh 2011) can produce samples from a probability density $p(x)$ using only the gradients $\nabla_x \log p(x)$ in a Markov chain of updates:
$$x_t = x_{t-1} + \frac{\delta}{2}\nabla_x \log p(x_{t-1}) + \sqrt{\delta}\epsilon_t, \text{ where } \epsilon_t \sim \mathcal{N}(0,\mathbf{I})$$
where $\delta$ is the step size. When $T \to \infty, \delta \to 0$, $x_T$ equals to the true probability density $p(x)$.
Compared to standard SGD, stochastic gradient Langevin dynamics injects Gaussian noise into the parameter updates to avoid collapses into local minima.\
"""

"""
# Reverse diffusion process

If we can reverse the above process and sample from $q(x_{t-1}|x_t)$, we will be able to recreate the true sample from a Gaussian noise input, $x_T \sim \mathcal{N}(0,\mathbf{I})$. Note that if $\beta_t$ is small enough, $q(x_{t-1}|x_t)$ will also be Gaussian. Unfortunately, we cannot easily estimate $q(x_{t-1}|x_t)$ because it needs to use the entire dataset and therefore we need to learn a model $p_\theta$ to approximate these conditional probabilities in order to run the *reverse diffusion process*.

$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$$
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$$

It is noteworthy that the reverse conditional probability is tractable when conditioned on $x_0$:

$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}(x_t,x_0), \tilde{\beta}_t\mathbf{I})$$

Using Bayes' rule, we have:

$$
\begin{aligned}
q(x_{t-1}|x_t,x_0) &= \frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)} \\
&\propto \exp(-\frac{1}{2}(\frac{(x_t-\alpha_tx_{t-1})^2}{\beta_t} + \frac{(x_{t-1}-\bar{\alpha}_{t-1}x_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(x_t-\bar{\alpha}_tx_0)^2}{1-\bar{\alpha}_t})) \\
&= \exp(-\frac{1}{2}(\frac{x_t^2-2\alpha_tx_tx_{t-1}+\alpha_tx_{t-1}^2}{\beta_t} + \frac{x_{t-1}^2-2\bar{\alpha}_{t-1}x_0x_{t-1}+\bar{\alpha}_{t-1}x_0^2}{1-\bar{\alpha}_{t-1}} - \frac{(x_t-\bar{\alpha}_tx_0)^2}{1-\bar{\alpha}_t})) \\
&= \exp(-\frac{1}{2}((\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}})x_{t-1}^2 - (\frac{2\alpha_t}{\beta_t}x_t+\frac{2\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}}x_0)x_{t-1} + C(x_t,x_0)))
\end{aligned}
$$

where $C(x_t,x_0)$ is some function not involving $x_{t-1}$ and details are omitted. Following the standard Gaussian density function, the mean and variance can be parameterized as follows (recall that $\alpha_t=1-\beta_t$ and $\bar{\alpha}_t=\prod_{i=1}^t \alpha_i$):

$$
\begin{aligned}
\tilde{\beta}_t &= 1/(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}) \\
&= 1/(\frac{\alpha_t-\bar{\alpha}_t+\beta_t}{\beta_t(1-\bar{\alpha}_{t-1})}) \\
&= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t
\end{aligned}
$$

$$
\begin{aligned}
\tilde{\mu}_t(x_t,x_0) &= (\frac{\alpha_t}{\beta_t}x_t+\frac{\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}}x_0)/(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}) \\
&= (\frac{\alpha_t}{\beta_t}x_t+\frac{\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}}x_0)\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t \\
&= \frac{\alpha_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+\frac{\bar{\alpha}_{t-1}\beta_t}{1-\bar{\alpha}_t}x_0
\end{aligned}
$$

Thanks to the nice property, we can represent $x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_t)$ and plug it into the above equation and obtain:

$$\tilde{\mu}_t = \frac{1}{\alpha_t}(x_{t-1}-\frac{\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_t)$$

[Continued in next message due to length...]

As demonstrated in Fig. 2., such a setup is very similar to VAE and thus we can use the variational lower bound to optimize the negative log-likelihood.

$$
\begin{aligned}
-\log p_\theta(x_0) &\leq -\log p_\theta(x_0) + D_{KL}(q(x_{1:T}|x_0)\|p_\theta(x_{1:T}|x_0)); \text{ KL is non-negative} \\
&= -\log p_\theta(x_0) + \mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}[\log \frac{q(x_{1:T}|x_0)p_\theta(x_{0:T})}{p_\theta(x_0)}] \\
&= -\log p_\theta(x_0) + \mathbb{E}_q[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0)] \\
&= \mathbb{E}_q[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}]
\end{aligned}
$$

Let $\mathcal{L}_{VLB} = \mathbb{E}_{q(x_{0:T})}[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}] \geq -\mathbb{E}_{q(x_0)}\log p_\theta(x_0)$

It is also straightforward to get the same result using Jensen's inequality. Say we want to minimize the cross entropy as the learning objective,

$$
\begin{aligned}
\mathcal{L}_{CE} &= -\mathbb{E}_{q(x_0)}\log p_\theta(x_0) \\
&= -\mathbb{E}_{q(x_0)}\log(\int p_\theta(x_{0:T})dx_{1:T}) \\
&= -\mathbb{E}_{q(x_0)}\log(\int \frac{q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)}p_\theta(x_{0:T})dx_{1:T}) \\
&= -\mathbb{E}_{q(x_0)}\log(\mathbb{E}_{q(x_{1:T}|x_0)}\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}) \\
&\leq -\mathbb{E}_{q(x_{0:T})}\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \\
&= \mathbb{E}_{q(x_{0:T})}[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}] = \mathcal{L}_{VLB}
\end{aligned}
$$

[Continued in next message...]

To convert each term in the equation to be analytically computable, the objective can be further rewritten to be a combination of several KL-divergence and entropy terms:

$$
\begin{aligned}
\mathcal{L}_{VLB} &= \mathbb{E}_{q(x_{0:T})}[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}] \\
&= \mathbb{E}_q[\log \frac{\prod_{t=1}^T q(x_t|x_{t-1})}{p_\theta(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)}] \\
&= \mathbb{E}_q[-\log p_\theta(x_T) + \sum_{t=1}^T \log \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)}] \\
&= \mathbb{E}_q[-\log p_\theta(x_T) + \sum_{t=2}^T \log \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)} + \log \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}] \\
&= \mathbb{E}_q[-\log p_\theta(x_T) + \sum_{t=2}^T \log \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)} \cdot \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} + \log \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}] \\
&= \mathbb{E}_q[D_{KL}(q(x_T|x_0)\|p_\theta(x_T)) + \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1)]
\end{aligned}
$$

Let's label each component in the variational lower bound loss separately:

$$\mathcal{L}_{VLB} = L_T + L_{T-1} + \cdots + L_0$$

where:
$$
\begin{aligned}
L_T &= D_{KL}(q(x_T|x_0)\|p_\theta(x_T)) \\
L_t &= D_{KL}(q(x_t|x_{t+1},x_0)\|p_\theta(x_t|x_{t+1})) \text{ for } 1 \leq t \leq T-1 \\
L_0 &= -\log p_\theta(x_0|x_1)
\end{aligned}
$$

Every KL term in $\mathcal{L}_{VLB}$ (except for $L_0$) compares two Gaussian distributions and therefore they can be computed in closed form. $L_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $x_T$ is a Gaussian noise. Ho et al. 2020 models $L_0$ using a separate discrete decoder derived from $\mathcal{N}(x_0; \mu_\theta(x_1,1), \Sigma_\theta(x_1,1))$.
""" -->
