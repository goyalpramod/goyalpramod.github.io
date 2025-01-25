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

## The Idea behind Stable Diffusion 


## Components of SD

As is the nature of Understanding Stable Diffusion, it is going to be mathematics heavy. I have added an appendix at the bottom where I explain each mathematical ideas as simply as possible. 

It will take too much time and distract us from the understanding of the topic being talked at hand if I describe the mathematical ideas as well as the idea of the process in the same space. 

## Maths of the Forward Diffusion process 

Imagine you have a large dataset of images, we will represent this real data distribution as $q(x)$ and we take an image from it (data point) $x_0$.
(Which is mathematically represented as  $x_0 \sim q(x)$).

In the forward diffusion process we add small amounts of Gaussian noise to the image ($x_0$) in $T$ steps. Which produces a bunch of noisy images as each step which we can label as $x_1,\ldots,x_T$. These steps are controlled by a variance schedule given by $\beta_t$. The value of $\beta_t$ ranges from 0 to 1 (i.e it can take values like 0.002, 0.5,0.283 etc) for $t, \ldots, T$. (Mathematically represented as ${\beta_t \in (0,1)}_{t=1}^T$)

There are many reasons we choose Gaussian noise, but it's mainly due to the properties of normal distribution. (about which you can read more here)

Now let us look at the big scary forward diffusion equation and understand what is going on

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I}) \tag{1}$$
$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1}) \tag{2}$$


$q(x_t|x_{t-1})$ means that given that I know $q(x_{t-1})$ what is the probability of $q(x_t)$ This is also knows as [bayes theorem](). 

To simplify it, think of it as. given $q(x_0)$ (for value of $t$ = 1) I know the value of $q(x_1)$.

The right handside of equation 1 represents a normal distribution. 

Now A question that I had was how can a probability and distribution be equal, well the Left Hand Side(LHS) of equation(eq) 1 represents a Probability Density Function ([PDF]())

For the Right Hand Side(RHS) of eq 1. When we write $N(x; μ, σ²)$, we're specifying that $x$ follows a normal distribution with mean $μ$ and variance $σ²$
 
This can be written as 

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

As  $t$ becomes larger. And eventually when $T \to \infty$ (This means as $T$ approaches infinity, or just a really large number). The initial data sample $x_0$ loses its features and turns into an isotropic Gaussian Distribution.

{explain equation 2 as well}


Let's talk about an interesting property - we can actually sample $x_t$ at any arbitrary time step. This means we don't need to go through the diffusion process step by step to get to a specific noise level.

First, let's understand something fundamental about normal distributions. Any normal distribution can be represented in the following form:

$$X = \mu + \sigma \epsilon$$

where $\epsilon \sim \mathcal{N}(0,1)$ (This means $\epsilon$ is sampled from a normal distribution with mean 0 and variance 1)

Taking our equation from before:
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$$

We can rewrite this using the above form as:
$$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_{t-1}$$

To make our equations simpler, let's define $\alpha_t = 1-\beta_t$. This gives us:
$$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}$$

Now, we can substitute the expression for $x_{t-1}$ in terms of $x_{t-2}$:
$$x_t = \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1}$$

A key property of normal distributions is that when we add two normal distributions, their means and variances can be combined. Using this property and some algebraic manipulation, we get:

$$x_t = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\bar{\epsilon}_{t-2}$$

If we continue this process all the way back to our original image $x_0$, and define $\bar{\alpha}_t$ as the product of all $\alpha$s from 1 to t ($\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$), we arrive at:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

This final equation is quite powerful. It allows us to directly sample $x_t$ at any timestep $t$ using just:
- The original image $x_0$
- The cumulative product of alphas up to time $t$ ($\bar{\alpha}_t$)
- A sample from a standard normal distribution ($\epsilon$)

This makes our implementation much more efficient as we can directly jump to any noise level without calculating all the intermediate steps.

{explain about alpha as well, and rewrite this in your tone a bit more}

"""\
Usually, we can afford a larger update step when the sample gets noisier, so $\beta_1 < \beta_2 < \cdots < \beta_T$ and therefore $\bar{\alpha}_1 > \cdots > \bar{\alpha}_T$.\
"""

"""\
**Connection with stochastic gradient Langevin dynamics**\
Langevin dynamics is a concept from physics, developed for statistically modeling molecular systems. Combined with stochastic gradient descent, stochastic gradient Langevin dynamics (Welling & Teh 2011) can produce samples from a probability density $p(x)$ using only the gradients $\nabla_x \log p(x)$ in a Markov chain of updates:
$$x_t = x_{t-1} + \frac{\delta}{2}\nabla_x \log p(x_{t-1}) + \sqrt{\delta}\epsilon_t, \text{ where } \epsilon_t \sim \mathcal{N}(0,\mathbf{I})$$
where $\delta$ is the step size. When $T \to \infty, \delta \to 0$, $x_T$ equals to the true probability density $p(x)$.
Compared to standard SGD, stochastic gradient Langevin dynamics injects Gaussian noise into the parameter updates to avoid collapses into local minima.\
"""

Let me help you understand Langevin dynamics and its connection to diffusion models. This is a fascinating bridge between physics and machine learning!
First, let's understand what Langevin dynamics is trying to do. Imagine you're trying to find the lowest point in a hilly landscape while blindfolded. If you only walk downhill (like regular gradient descent), you might get stuck in a small valley that isn't actually the lowest point. Langevin dynamics solves this by occasionally taking random steps - like sometimes walking uphill - which helps you explore more of the landscape.
The key equation is:
xt = xt−1 + (δ/2)∇xlog⁡p(xt−1) + δϵt
Let's break this down piece by piece:

xt and xt−1 represent our position at the current and previous steps
∇xlog⁡p(xt−1) is the gradient term - it tells us which direction to move to increase the probability
δ is our step size - how far we move in each step
ϵt is our random noise term, sampled from a normal distribution

The equation combines two behaviors:

A "deterministic" part: (δ/2)∇xlog⁡p(xt−1) which moves us toward higher probability regions
A "random" part: δϵt which adds noise to help us explore

What makes this special is that when we run this process for a long time (T→∞) and with very small steps (δ→0), we're guaranteed to sample from the true probability distribution p(x). This is similar to how diffusion models gradually denoise images - they're following a similar kind of path, but in reverse!
The connection to standard gradient descent is interesting - regular SGD would only have the gradient term, but Langevin dynamics adds that noise term ϵt. This noise prevents us from getting stuck in bad local minima, just like how shaking a jar of marbles helps them settle into a better arrangement.

Let me explain why we use the logarithm of the probability (log p(x)) instead of just the probability p(x) itself. This is a fundamental concept in probability and optimization that's used across many areas of machine learning.
First, let's understand why logarithms are useful in general with probabilities:

Numerical Stability
When we work with probabilities, we're dealing with numbers between 0 and 1. When we multiply many small probabilities together (which happens often in machine learning), we can end up with extremely small numbers that computers struggle to handle accurately. For example:
0.01 × 0.01 × 0.01 = 0.000001

But if we take logarithms:
log(0.01) + log(0.01) + log(0.01) = -13.82
The logarithm transforms these tiny numbers into more manageable negative numbers that computers can handle better.

Converting Multiplication to Addition
In probability theory, we often need to multiply many probabilities together. Taking the logarithm converts multiplication into addition, which is computationally simpler and more stable:
log(a × b) = log(a) + log(b)
Gradient Properties
Now, here's the key insight for why we use log p(x) in Langevin dynamics: When we take the gradient of log probability (∇log p(x)), we get something called the "score function". This score function has a special property - it points in the direction where the probability increases most rapidly.

Let's see why through calculus:
∇log p(x) = ∇(log p(x)) = (1/p(x))∇p(x)
This division by p(x) acts as an automatic scaling factor. When p(x) is small, it makes the gradient larger, and when p(x) is large, it makes the gradient smaller. This natural scaling helps our sampling process explore the probability space more efficiently.

What is P(x) though and why are we taking that. Traditionally in SGD do we not take, del(error)/del(weight)

In traditional SGD for neural networks, we're trying to minimize an error function (or loss function), so we use ∂(error)/∂(weight) as you correctly noted. We're trying to find the weights that make our predictions as accurate as possible.
But in Langevin dynamics, we're doing something fundamentally different. Here, p(x) represents a probability distribution that we want to sample from. Think of it this way:
Imagine you have a dataset of faces, and you want to generate new faces that look real. The probability p(x) would represent how likely it is that a particular image x is a real face. Areas of high p(x) would correspond to images that look like real faces, while areas of low p(x) would be images that don't look like faces at all.
So when we take ∇log p(x), we're asking: "In which direction should I move to make this image look more like a real face?"
Let's compare:

Traditional SGD in Neural Networks:

Goal: Minimize error
Gradient: ∂(error)/∂(weight)
Updates weights to make predictions more accurate


Langevin Dynamics:

Goal: Sample from a probability distribution
Gradient: ∇log p(x)
Updates the sample itself to look more like real data



This is why Langevin dynamics is particularly relevant to diffusion models. Remember how diffusion models start with noise and gradually transform it into an image? The ∇log p(x) term tells us how to modify our noisy image at each step to make it look more like real data.

"""

TRAINING THE MODEL

Since we can't have x₀ during generation, we train a model pθ(xₜ₋₁|xₜ) to approximate q(xₜ₋₁|xₜ,x₀). This model learns to predict the denoising step without needing the original image.
The training process works like this:

Take a clean image x₀
Sample a random timestep t
Add noise to get xₜ using our "nice property" formula
Train the model to predict the noise that was added
The model learns to do this by minimizing the difference between its prediction and the actual noise

# Maths of Reverse diffusion process
Now what we want to do is take a noisy image $x_t$ and get the original image $x_0$ from it. And to do that we need to do a reverse diffusion process. 

Essentially we want to sample from $q(x_{t-1}|x_t)$, Which is quite tough as there can be millions of noisy images for actual images. To combat this we create an approximation (why do they work and how do they work in a minute) $p_\theta$ to approximate these conditional probabilities in order to run the *reverse diffusion process*.

Which can be represented as 
$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$$
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$$

Unfortunately it is tough to even sample from this approximate model because it is the same as our previous model, so we modify it by adding the original image $x_0$ to it as such. 

$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}(x_t,x_0), \tilde{\beta}_t\mathbf{I})$$

Now this is tractable (Exaplain what this word means), let us first understand the proof for how it is tractable. Later moving on to understand how they thought of this idea in the first place 


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

{add color coding for the above to make it easier to understand}


"""
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
"""



### Unet 

You will be surprised to know that the idea of Unets comes from a [medical paper](https://arxiv.org/pdf/1505.04597)


As is the nature of things, I am going with the assumption you understand what CNNs are 
and the different kinds of task in image classification. Below I have created a simple visualization for the different tasks. 

But if you do not know CNNs Let me give a quick overview below otherwise, consider reading my CV blog(If the link doesn't take you anywhere, it's still under development) or read this.

Helpful docs 

[Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)\
[BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html1)\
[MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)\
[Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)\
[torch.cat](https://pytorch.org/docs/main/generated/torch.cat.html)\
[ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)\
[Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)




### VAE 


### CLIP

### DDPM 



### DDIM

## The code 

## Understanding the metrics 

This is interesting as well because... how do you tell a computer which is a good image and which is a bad image without actually doing a vibe check.

This really makes you appreaciate how the loss function was created doesnt it now!!

## Things to talk about from the fast ai notebooks:
* [Stable diffsion components](https://forbo7.github.io/forblog/posts/13_implementing_stable_diffusion_from_its_components.html) Build SD from taking components from HF
* 

## Appendix

### PDF 

### KL Divergence

### Bayes' rule -->