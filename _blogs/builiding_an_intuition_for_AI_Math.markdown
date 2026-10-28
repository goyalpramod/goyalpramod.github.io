<!-- ---
layout: blog
title: "Building an Intuition for ML Math"
date: 2026-09-25 12:00:00 +0530
categories: [personal, technology]
image: assets/blog_assets/supe_fast_inference/b20vm4.jpeg
---

"ML MATH IS NOT HARD, YOU HAVE JUST BEEN TAUGHT WRONG"

One of the hardest pieces of math that I have ever encountered in machine learning was while learning diffusion. At that time, I saw a bunch of tutorials, read a few blogs, burned lots of tokens, and wrote about it in a blog.

At that time I deeply believed that I understood everything I wrote about. But as I re-read it and reflect on it, I realise that my understanding was shallow at best and superficial at worst. 

I am now a bit older and perhaps a bit wiser, so I think it is better that we work towards building an intuition for ML mathematics instead of just understanding one piece of it.

> I write this while preparing for my mid-term on Generative AI (MSAI632) which is in less than a week, so ideally I should wrap this up before that. Also at this time I am quite occupied with other responsibilities, but that does not mean I will sacrifice the sanctity of this work.

Let us begin. 

## The equations we will be breaking down 

I believe it will be a good objective to take some of the popular ML works and break down their mathematics to understand how it came to be, what the techniques are, and truly grasp what goes on!

* Transformers
* VAE
* GANs
* Diffusion
* Flow Matching
* GRPO 

## Understanding the objective 

The first thing to do while dealing with any ML problem is to understand what the objective of that particular work is.

By now I am sure you are aware that there are essentially three brackets under which most (if not all) machine learning problems fall.

1. Supervised learning -> Given data, make a function that generalizes on that data and predicts on unknown data. It can simply be written as 

$$
y = f(x)
$$

and $f(\cdot)$ is what we are trying to learn. We have labelled examples $(x, y)$, using which we try to predict $y$ for unseen $x$.

2. Self-Supervised learning -> Given data, use that data to learn patterns in the data. The best example would be pretraining of LLMs.

It can be crudely represented as

$$
P(x_{i+1} \mid x_1, x_2, \dots, x_i)
$$

3. Reinforcement learning -> Given a reward function and an environment to act on with actions, maximize the cumulative reward.

It can be crudely represented as 

$$
q_\pi(s, a) \doteq \mathbb{E}_\pi\left[G_t \mid S_t = s, A_t = a\right]
$$

Where $G_t$ is the return, the (discounted) sum of all the rewards we collect from time $t$ onwards.

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

(P.S. I am actively writing more about RL, check my blog series here)

## Transformers

*The equation we are heading towards: Eq. 1 of Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017)*

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$


This is probably going to be the simplest of the equations we will be dealing with today. But I wanted to include it because in most ML works that you will encounter, there will be some form of matrix multiplication going on, and looking at the most popular one can be a good first step, imo.

$QK^\top$ is just a simple matrix multiplication ($K^\top$ is $K$ transposed).

For a matrix $A$ of shape $M \times K$ multiplied with a matrix $B$ of shape $K \times N$ we get a matrix of shape $M \times N$. The inner dimensions must match for the multiplication to be successful.

We divide that by $\sqrt{d_k}$ to keep the numbers well behaved, and it is kind of intuitive to think why as well. Each entry of $QK^\top$ is a dot product of two vectors of length $d_k$, which means it is a sum of $d_k$ terms. The bigger $d_k$ is, the bigger these sums tend to get (if every entry has a mean of 0 and a variance of 1, the dot product ends up with a variance of $d_k$). Feed very large numbers into a softmax and it collapses to something that is almost one-hot, where the gradients all but vanish. Dividing by $\sqrt{d_k}$ brings the variance back down to 1, which stabilizes the computation by normalizing it.

We then do a softmax. 

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Softmax is pretty straightforward. It takes a set of numbers and turns them into a probability distribution: every value becomes positive and they all add up to 1 (in attention it is applied to each row of $\frac{QK^\top}{\sqrt{d_k}}$ separately).

We then do a matrix multiplication with $V$, which finally gives us the attention: every output is a weighted average of the rows of $V$, where the weights are the softmax scores.

The image below captures the whole idea in essence.


[insert_image_from_transformers_blog]

> Image taken from my blog on transformers, consider reading it to understand it more. 

## VAE

*The equations we are heading towards: Section 2.2 (Eqs. 1-3) of Kingma & Welling, ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) (2013)*

$$
\log p_\theta(\mathbf{x}^{(i)}) = D_{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}^{(i)}) \,\|\, p_\theta(\mathbf{z} \mid \mathbf{x}^{(i)})\big) + \mathcal{L}(\theta, \phi; \mathbf{x}^{(i)}) \tag{1}
$$

$$
\log p_\theta(\mathbf{x}^{(i)}) \geq \mathcal{L}(\theta, \phi; \mathbf{x}^{(i)}) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\left[-\log q_\phi(\mathbf{z} \mid \mathbf{x}) + \log p_\theta(\mathbf{x}, \mathbf{z})\right] \tag{2}
$$

$$
\mathcal{L}(\theta, \phi; \mathbf{x}^{(i)}) = -D_{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}^{(i)}) \,\|\, p_\theta(\mathbf{z})\big) + \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x}^{(i)})}\left[\log p_\theta(\mathbf{x}^{(i)} \mid \mathbf{z})\right] \tag{3}
$$

and the "naive" gradient estimator the paper rejects because of its high variance (the problem the reparameterization trick solves)

$$
\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z})}\left[f(\mathbf{z})\right] = \mathbb{E}_{q_\phi(\mathbf{z})}\left[f(\mathbf{z}) \nabla_{q_\phi(\mathbf{z})} \log q_\phi(\mathbf{z})\right] \simeq \frac{1}{L}\sum_{l=1}^{L} f(\mathbf{z}) \nabla_{q_\phi(\mathbf{z}^{(l)})} \log q_\phi(\mathbf{z}^{(l)}), \quad \mathbf{z}^{(l)} \sim q_\phi(\mathbf{z} \mid \mathbf{x}^{(i)})
$$

Now this is where things start to get a bit tougher, because we are going to dive into statistics and probability. Earlier we had a look at linear algebra.

In linear algebra I skipped over a bunch of ideas because they are pretty rudimentary in my opinion. But that is not the case here, so let us look at a few things. 

The first thing we need to understand is that there are different kinds of distributions, all of which can be described by a probability function (a probability density function for continuous distributions). I know, these are all big words, but let me simplify it step by step.

Let's say you take a die, roll it 100 times, count the number of times each value occurred, and plot it.

It will most likely look something like this 

[CREATE_PLOT]

Now this won't be exactly equal (because real life does not work that way), so if I ask you, if I roll a die, what is the probability of 1 occurring, you can just do

`Favourable occurrences / total occurrences`

so in this case it will be roughly 1/6.

and that will be the case for all numbers in fact. Now this introduces us to two facts. First, the outcomes of a die roll are countable, you can count them one after the other. Hence they are `discrete` in mathematical terms. Additionally, as all numbers are equally likely to occur we can say that this is a uniform distribution (just one kind of distribution).

This lets us define the probability mass function for this distribution 

$$
P(X = k) = \frac{1}{6}, \quad k \in \{1, 2, 3, 4, 5, 6\}
$$

More generally, a probability mass function gives a probability to every possible value, and all of those probabilities must add up to 1

$$
p(x) = P(X = x), \qquad \sum_x p(x) = 1
$$

Similarly, let's say that you are measuring the temperature outside throughout the day.

It will look something like this

[DRAW_DIAGRAM]

This is called a normal distribution, and we can define it using two parameters, the mean $\mu$ and the variance $\sigma^2$. (In our earlier case, the probability was a constant for all values!) Now notice how you can not count the exact temperature nor the exact time, because this is a continuous distribution. So we must define a probability density function for it.

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

which is often written as $x \sim \mathcal{N}(\mu, \sigma^2)$. Note that $f(x)$ is a density and not a probability itself, the probability of landing between $a$ and $b$ is the area under the curve, $\int_a^b f(x)\,dx$.

The PDF is often used in AI when we are sampling (picking) a specific value. Now if you want the sum of probability up to a value of $x$, we get that by integrating the PDF, which gives us the Cumulative Distribution Function, or CDF (for discrete distributions we just add up the PMF instead of integrating).

So for instance if we want $P(x \le X)$ we will write it as

$$
F(X) = P(x \le X) = \int_{-\infty}^{X} f(x)\, dx
$$

Now there are many other kinds of distributions, each with their own parameters that can be used to define them, like

**Poisson**: the number of events in a fixed interval, where $\lambda$ is the average rate of events

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots
$$

**Bernoulli**: a single coin flip, where $p$ is the probability of getting a 1

$$
P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

**Multivariate Gaussian**: the normal distribution extended to $d$ dimensions, with a mean vector $\boldsymbol{\mu}$ and a covariance matrix $\Sigma$

$$
f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d \, |\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

Ok, hopefully we now understand distributions and a few things about them. Now we can talk about a slightly more complex idea: expectation and variance. Just a complex way of saying what we already talked about.

The expectation of a distribution is just its mean, and the variance... is well the variance. (Another helpful thing to remember is that the mean is what the distribution is centered around, whereas the variance is the spread of the data, or how spread out the distribution is. For a normal distribution the mean is also the most likely value, but that is not true for every distribution, the expectation of a die roll is 3.5, a value you can never actually roll!)

There are some helpful properties of expectation and variance that we will use and I believe it will be helpful to remember

$$
\mathbb{E}[X] = \sum_x x\, p(x) \qquad \text{or} \qquad \mathbb{E}[X] = \int x\, f(x)\, dx
$$

$$
\text{Var}(X) = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - \left(\mathbb{E}[X]\right)^2
$$

and the properties we will lean on (here $a$ and $b$ are constants)

$$
\begin{aligned}
\mathbb{E}[aX + b] &= a\,\mathbb{E}[X] + b \\
\mathbb{E}[X + Y] &= \mathbb{E}[X] + \mathbb{E}[Y] \\
\text{Var}(aX + b) &= a^2\, \text{Var}(X) \\
\text{Var}(X + Y) &= \text{Var}(X) + \text{Var}(Y) \quad \text{if } X \text{ and } Y \text{ are independent}
\end{aligned}
$$

Finally, the expectation of a function of $X$, which shows up all the time when we write losses

$$
\mathbb{E}[g(X)] = \sum_x g(x)\, p(x) \qquad \text{or} \qquad \mathbb{E}[g(X)] = \int g(x)\, f(x)\, dx
$$

I remember my math professor telling us something that will be helpful. "If someone told you that you do not need to remember things in math and you can always derive them, they lied to you."

We are not done yet. We have to talk about another idea before we can start talking about VAEs, and that is the idea of Bayesian statistics (this is a huge field of mathematics on its own, but I like the Pareto principle of 80/20: essentially you only need to know 20% of things to get 80% of the work done, heck that can be said about this whole article). The formula looks something like this

$$
p(a \mid b) = \frac{p(b \mid a)\, p(a)}{p(b)}
$$

The left side is called the posterior, the right side is made up of prior (), likelihood (), and evidence/marginal probability. All very long, weird, complex math words. But we still need nomenclature to talk amongst ourselves. 

But they do make sense if you look at them again 

[] is the prior, it essentially means what is our prior belief
[] is the likelihood, it essentially means, given that we know the item on the right is true, what are the chances of the item on the left being true
[] is the marginal, essentially the overall chance of this happening across all possible cases

[ADD_THE_DIAGRAM_OF_THE_VENN_DIAGRAM]

This is also called conditional probability, because it is defined as `p(a|b)`: if b is true, what are the chances of a being true.

Another thing that we also need to talk about is joint probabilities (I know these might seem overwhelming if it is your first time learning about all these. But they are important concepts you will have to deal with daily if you decide to pursue deeper work in ML/AI)

`p(a,b)`

It essentially means what is the probability of both a and b happening simultaneously. We can write that as

$$
p(a, b) = p(b)\, p(a \mid b)
$$

Now we already know what this means, say it with me. The probability of b happening multiplied with the probability of a happening given that b happened.

[ADD THE LINK TO VISUAL INFORMATION THEORY AS THAT EXPLAINS ALL OF THIS IN GREAT DETAIL]

Now if you go back and look at the original equation, some of the ideas may seem less intimidating than before, but a bit scary still. Let's change that by talking about the final things you need to learn.

The two final ideas we need to talk about are entropy and KL Divergence.

Now Christopher Olah does a phenomenal job explaining these ideas in his blog, which I insist you check out. Check them out here, no seriously check them out.

Entropy is essentially the idea of how many bits are required, on average, to encode an outcome drawn from a given distribution.

For instance if you just want to represent the numbers 0 and 1, you can represent them using one bit as either 0 or 1. If you want to represent 0, 1, 2 and 3, you can do it using 2 bits as 00, 01, 10 and 11. (In general, $n$ bits can represent $2^n$ different values, so 2 bits give us $2^2 = 4$ values.)

And we can define the entropy of any given distribution as

$$
H(p) = -\sum_x p(x) \log_2 p(x)
$$

If you wish to understand how that works, seriously check this blog out. But in a nutshell

{add explanation}

Ok now that you understand what is going on, we need to talk about KL divergence. It is essentially the idea of what happens if a distribution $p$ is represented by a distribution $q$: how many extra bits we pay when we encode samples from $p$ using a code built for $q$. Or in simpler terms, how different the two distributions are (it is not symmetric, so it is not a true distance).

Which can mathematically be represented as

$$
D_{KL}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

where $H(p, q) = -\sum_x p(x) \log q(x)$ is the cross-entropy, the average number of bits needed to encode samples from $p$ using a code built for $q$.

And that, surprisingly, is all the math that you need to understand to get this done (with a few caveats and explanations that we will deal with as we move forward).

Now let us come back to VAEs, my objective here is not exactly to explain how VAEs work but more about helping you understand the landscape behind the math for it. 

If you would like to gain a better and deeper understanding of them, consider checking these blogs out 

[ADD_REFERENCES]

[ADD_VAE_DIAGRAM]

Now in simple terms we have an encoder, a bottleneck, and a decoder 

[AI_WRITE_THE_MATH_IN_CONDITIONAL_PROBABILITY]

https://mbernste.github.io/posts/variational_inference/
https://mbernste.github.io/posts/vae/
https://mbernste.github.io/posts/elbo/

Much more fulfing blog on Probability and Measure thoery -> https://mbernste.github.io/posts/measure_theory_1/

[Explain it better above] [Or add link to jeremys blog and recommend reading that]

https://blog.evjang.com/2016/08/variational-bayes.html -> Beautiful derivation!!!

## GANs

*The equation we are heading towards: Eq. 1 of Goodfellow et al., ["Generative Adversarial Nets"](https://arxiv.org/abs/1406.2661) (2014)*

$$
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}\left[\log D(\mathbf{x})\right] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})}\left[\log\left(1 - D(G(\mathbf{z}))\right)\right]
$$

## Diffusion

*The derivation we are heading towards: Appendix A (Eqs. 17-22) of Ho et al., ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) (2020), reducing the variational bound to a sum of KL terms*

$$
\begin{aligned}
L &= \mathbb{E}_q\left[-\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}\right] && (17) \\
&= \mathbb{E}_q\left[-\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}\right] && (18) \\
&= \mathbb{E}_q\left[-\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})} - \log \frac{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}{q(\mathbf{x}_1 \mid \mathbf{x}_0)}\right] && (19) \\
&= \mathbb{E}_q\left[-\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)} \cdot \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)} - \log \frac{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}{q(\mathbf{x}_1 \mid \mathbf{x}_0)}\right] && (20) \\
&= \mathbb{E}_q\left[-\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T \mid \mathbf{x}_0)} - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)} - \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)\right] && (21) \\
&= \mathbb{E}_q\left[D_{KL}\big(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, p(\mathbf{x}_T)\big) + \sum_{t > 1} D_{KL}\big(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big) - \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)\right] && (22)
\end{aligned}
$$


## Flow Matching

*The equations we are heading towards: Eqs. 5 and 9 of Lipman et al., ["Flow Matching for Generative Modeling"](https://arxiv.org/abs/2210.02747) (2022)*

$$
\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t,\, p_t(x)} \left\| v_t(x) - u_t(x) \right\|^2
$$

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t,\, q(x_1),\, p_t(x \mid x_1)} \left\| v_t(x) - u_t(x \mid x_1) \right\|^2
$$

## GRPO

*The equation we are heading towards: Eq. 3 of Shao et al., ["DeepSeekMath"](https://arxiv.org/abs/2402.03300) (2024)*

$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) &= \mathbb{E}\left[q \sim P(Q),\, \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O \mid q)\right] \\
&\quad \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Bigg\{ \min\left[ \frac{\pi_\theta(o_{i,t} \mid q, o_{i, < t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i, < t})} \hat{A}_{i,t},\ \text{clip}\left(\frac{\pi_\theta(o_{i,t} \mid q, o_{i, < t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i, < t})},\, 1-\varepsilon,\, 1+\varepsilon\right) \hat{A}_{i,t} \right] - \beta\, \mathbb{D}_{KL}\left[\pi_\theta \,\|\, \pi_{ref}\right] \Bigg\}
\end{aligned}
\tag{3}
$$

References ->
* https://www.jeremyjordan.me/autoencoders/
* https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/?ref=jeremyjordan.me
* https://www.jeremyjordan.me/variational-autoencoders/
* https://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in-prose-and-code.html
* https://gokererdogan.github.io/index.html
* https://jaan.io/what-is-variational-autoencoder-vae-tutorial#the-probability-model-perspective
* https://gokererdogan.github.io/2017/08/15/variational-autoencoder-explained/

Maths
* https://www.countbayesie.com/blog/2015/2/18/bayes-theorem-with-lego
* https://colah.github.io/posts/2015-09-Visual-Information/ -->
