<!-- ---
layout: blog
title: "Evolution of LLMs"
date: 2025-05-05 12:00:00 +0530
categories: [personal, technology]
image: assets/blog_assets/demystifying_diffusion_models/temp_meme_img.webp
---

[TOP MEME, MOnkey to man evolution yearwise and leave the man with no model (to signify no AGI yet)]
[Add tree from this paper to clarify that it was not a linear evolution but a tree wise, much like real theory of evolution https://arxiv.org/pdf/2304.13712]

The landscape of language models has evolved dramatically since the introduction of the Transformer architecture in 2017. Here we explore the

- mathematical foundations
- architectural innovations
- training breakthroughs

From attention mechanisms to constitutional AI, we'll dive deep into the code, math, and ideas that revolutionized NLP.

Additionally you can treat this blog as a sort of part 2, to my original blog on transformers which you can checkout [here](https://goyalpramod.github.io/blogs/Transformers_laid_out/).

## How this blog is structured

First we would begin with a short to the transformers (Most LLM structure begin this way), then go sequentially how they are trained and infererenced. To lay a groundwork, on top of which we can build.

After which we will have a look at a short AI timeline over the years, going year by year seeing the most influential work that shaped that and future years of LLMs.

Where we will see what architectural innovation, computational breakthrough, training optimization that were invented over the years and how it affected LLMs and their benchmarks.

Finally and most importantly we will dive deep into the technical understanding and implementation of these different techniques. Some of them being Flash Attention, KV-Caching, GRPO etc.

Additionally there have been a lot of innovations in vision modeling, TTS, Image gen, Video gen etc each of which deserves it's own blog(And there will be!! I promise you that). Over here I will just give quick intro and links to some ground breaking innovations.

> NOTE: Do not take for granted all the hardware, data and benchmark innovations, Though I will briefly mention them in the timeline. I implore you to explore them further if they interest you. This blog is strictly restricted to breakthroughs in Large Language Models, and mostly open source one's. Even though current models by OpenAI are amazing, not much is known about them to the public. So we will briefly talk about what we know about them, then move on to talk about mostly open source models.

## A short introduction to LLMs

This part is highly influenced by this [video](https://www.youtube.com/watch?v=7xTGNNLPyMI) by andrej karpathy

A paper on pretraining [paper](https://arxiv.org/pdf/2003.08271)

### Architecture

We will be skipping over the internal details about the transformers model (you can read more about it in my previous blog), I will proceed with the assumption that you have a very deep level understanding of atleast the transformer model. Having that that let us proceed.

As we can see the original transformer has two parts, an Encoder and a Decoder. And as is known, it was initially made for the sole purpose of Machine Translation.

But over the years they have been used for a plethora of tasks from

- Question Answering
- Summarization
- Tagging
- Classification

And many more.

LLMs consist of architectures which are solely based on the Encoder like Bert

[ADD_IMAGE_OF_BERT]

LLMs consist of architectures which are solely based on the Decoder like gpt-1

[ADD_IMAGE_OF_GPT1]

There are LLMs which use the good ol Encoder Decoder layer together too like T5

[WITH_IMAGE]

One thing to be mindful of is, that the development of LLMs have been done on the transformers. There was no radical shift, only gradual and slow incremental changes. As you read more about them you will understand them better.

In crux you only need to understand the basic Transformer architecture to understand most if not all LLMs.

### Training

As I mentioned there are 3 kinds of architectures when it comes to LLM, hence there is a different way of training each. Even though they share few similarities. Each has a different objective. Let us begin by first understanding the most popular LLM architecture, that being the decoder only architecture.

#### Decoder Only

##### Pretraining

**Step 1: Data Collection**

We know Decoder based LLMs are able to answer a variety of answer to questions about science, food, mathematics, facts etc. That is because they have been trained on a huge plethora of the data.

So the first step is to collect the said data. One important thing to keep in mind is that these models are able to answer based on what they have seen. They are not performing any kind of reasoning. It is given a distribution, what is the most likely token that is to appear.

So if you want a coding LLM you will collect a lot of code related data from publically available places like github.

If you want a cooking LLM you will collected a lot of recipes and so on.

Most general purpose LLMs are trained on data collected from various sources, hence they are able to answer a lot of question.

A lot of filtering steps also goes behind it

"
Garbage in, garbage out
"

Most of the internet when crawled has data which looks something like this

```html
<html>
  {add shit}
</html>
```

Hence it needs to be processed into a more human readable form, You can imagine how humongous of a task it must be to clean huge datasets with GBs of data.

Now there are other filterning that also needs to be done, How do you take care of profanity? What about fake news and so on

**Step 2: Tokenization**

{maybe talk about vocabulary and token length}

When we talked about transformers, we skipped talking about tokenization, but as it is a vital piece of LLM training. We shall spend some time talking about it here.

Sentence level tokenization

My first question was, why do we need to break down words. Why not give the entire sentence. Heck why not give the whole paragraph as an input.

While in practice it can be done, it is not wise. Because if we go back to our basic principle about LLMs.

"They are next token predictors"

So in given any large enough paragraph or even an essay. The likelihood of a sentence repeating is very low. So if we think it in terms of machines, if we transform each sentence into a token, we will have a vocabular with a lot of numbers, which do not relate to each other at all. So we can never predict what sentence will come after any given sentence.

Word level tokenization

A simple work around this seems to be, well why not just tokenize the words, instead of sentences. Because in any large enough paragraph or essay. Words repeat and they follow a logical sequence of what is to appear next. For example

"Water is \_\_\_", if you gave me this sentence word by word, I will assume the next word is wet. Whereas if you gave me a sentence

"Water is wet, This is a debatable topic." I will have no clue what can be said after this sentence, Maybe someone raises a point, maybe someone says something else.

So word level helps us retain the logical sequence, and words have meanings to them too. But there is still one big issue. There can be millions of words, some have way higher representation in usual text and some are highly unlikely to occur in common place.

There are words which are commonplace in one industry and rare in another.

So we will have a huge vocabulary.

If you think for a moment we may come to the conclusion that, why not use character level tokenization to solve this problem, this will reduce the vocabulary drastically.

Here the problem would lie in the fact that characters by themselves do not hold much meaning (atleast in the english lexicon)

**Step 3: Training the network**

##### Fine-Tuning

#### Encoder Only

Now let's understand how a usual Encoder is trained, We will talk about BERT here

#### Encoder Decoder

Now let's do the same for T5

### Inference

## The AI timeline

This is a very short timeline of the most influential work, to read about more architectures that were huge at the time but died down eventually, consider going through the [Transformer catalog](https://docs.google.com/spreadsheets/d/1ltyrAB6BL29cOv2fSpNQnnq2vbX8UrHl47d7FkIf6t4/edit?gid=0#gid=0).

The blog ["Transformer models: an introduction and catalog — 2023 Edition"](https://amatria.in/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/) helped me immensely while making the timeline.

[Write the name of the creators and labs]

## 2017: The Foundation Year

### Transformer

[Attention is all you need](https://arxiv.org/abs/1706.03762)

The foundational paper on transformers , introduced some key ideas such as

- Scaled dot-product attention
- Multi-head attention mechanism
- Positional encodings
- Layer normalization
- Masked attention for autoregressive models

We have talked deeply about each of these topics previously and I implore you to check that part out [here]()

**Training a Transformer**

This is one topic that we didnt talk about extensively so let's go over it, because that is where the true beauty of GPT lies. How to train over huge amounts of data.

### RLHF - Reinforcement Learning from Human Preferences

[Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)

As mind boggling as it sounds, the famed algorithm RLHF came out in 2017, the same year attention is all you need came out.

Let us understand the ideas put forth and why it was such a big deal.

If you are new to RL, check it out in the [appendix]()

<details>
<summary>Quick Summary</summary>
"""
Quick Summary

This 2017 paper presents a method for training reinforcement learning (RL) agents using human feedback instead of explicitly defined reward functions. Here's a high-level overview:

The authors address a fundamental challenge in RL: for many complex tasks, designing appropriate reward functions is difficult or impossible. Instead of requiring engineers to craft these functions, they develop a system where:

1. Humans compare short video clips of agent behavior (1-2 seconds)
2. These comparisons train a reward predictor model
3. The agent optimizes its policy using this learned reward function

Key contributions:

- They show this approach can solve complex RL tasks using feedback on less than 1% of the agent's interactions
- This dramatically reduces the human oversight required, making it practical for state-of-the-art RL systems
- They demonstrate training novel behaviors with just about an hour of human time
- Their approach works across domains including Atari games and simulated robot locomotion

The technique represents a significant advance in aligning AI systems with human preferences, addressing concerns about misalignment between AI objectives and human values. By having humans evaluate agent behavior directly, the system learns rewards that better capture what humans actually want.
"""

</details>

**Problem** : Training a RL system requires researchers make a well define reward system, Which grows with complexity of system, Making it infeasible to train large RL systems

[Add image below, left side simple puzzle, right side complex puzzle]

**Solution** :

"An alternative approach is to allow a human to provide feedback on our system’s current behavior
and to use this feedback to define the task. In principle this fits within the paradigm of reinforcement
learning, but using human feedback directly as a reward function is prohibitively expensive for RL
systems that require hundreds or thousands of hours of experience."

[Show image of a man sitting tirelessly through 1000 of hours of RL]

"In summary, we desire a solution to sequential decision problems without a well-specified reward
function that

1. enables us to solve tasks for which we can only recognize the desired behavior, but not
   necessarily demonstrate it,
2. allows agents to be taught by non-expert users,
3. scales to large problems, and
4. is economical with user feedback"

"
We ask the human to
compare short video clips of the agent’s behavior, rather than to supply an absolute numerical
score. We found comparisons to be easier for humans to provide in some domains, while being
equally useful for learning human preferences.
Comparing short video clips is nearly as fast as
comparing individual states
"

[ADD Image from the paper]

[Add 2 image,

1. Human only shown parts of the way the model solving the problem
2. Only compares which approach is better
   ]

Now let us understand how a model, learns from these preferences. I.e the reward modeling

"""
**Reward Modeling in RLHF**


Read the following blogs to understand these topics better then explain them 
- https://huggingface.co/blog/rlhf
- https://huyenchip.com/2023/05/02/rlhf.html

The preference predictor model estimates the probability that a human would prefer trajectory segment σ¹ over σ²:

$$
\hat{P}\left[\sigma^{1} \succ \sigma^{2}\right]=\frac{\exp \sum \hat{r}\left(o_{t}^{1}, a_{t}^{1}\right)}{\exp \sum \hat{r}\left(o_{t}^{1}, a_{t}^{1}\right)+\exp \sum \hat{r}\left(o_{t}^{2}, a_{t}^{2}\right)}
$$

The reward function is trained using cross-entropy loss to match human preferences:

$$
\operatorname{loss}(\hat{r})=-\sum_{\left(\sigma^{1}, \sigma^{2}, \mu\right) \in D} \mu(1) \log \hat{P}\left[\sigma^{1} \succ \sigma^{2}\right]+\mu(2) \log \hat{P}\left[\sigma^{2} \succ \sigma^{1}\right]
$$

<details>
<summary>Mathematical Notation</summary>

- $\hat{P}\left[\sigma^{1} \succ \sigma^{2}\right]$: Predicted probability that trajectory segment $\sigma^{1}$ is preferred over trajectory segment $\sigma^{2}$
- $\hat{r}$: The learned reward function
- $o_{t}^{i}$: Observation at time $t$ in trajectory segment $i$
- $a_{t}^{i}$: Action at time $t$ in trajectory segment $i$
- $\sigma^{i}$: Trajectory segment $i$ (a sequence of observation-action pairs)
- $\exp$: Exponential function
- $\sum$: Summation over all timesteps in the trajectory segment
- $\operatorname{loss}(\hat{r})$: Cross-entropy loss function for the reward model
- $D$: Dataset of human preference comparisons
- $\mu$: Distribution over $\{1,2\}$ indicating human preference
- $\mu(1)$: Probability that human preferred segment 1
- $\mu(2)$: Probability that human preferred segment 2
- $\log$: Natural logarithm
</details>
""""
"""
Understanding the Reward Function Fitting Process

Let me break down this section step by step, which explains how the researchers train their reward function from human preferences:

1. The Preference-Predictor Model

The authors interpret their reward function estimate r̂ as a preference predictor. Instead of directly modeling a reward function, they model the probability that a human would prefer one trajectory segment over another.

2. The Mathematical Formulation (Equation 1)

The equation P̂[σ¹ ≻ σ²] represents the predicted probability that a human would prefer trajectory segment σ¹ over segment σ².

Breaking down the formula:
- σ¹ and σ² are two different trajectory segments (short video clips of agent behavior)
- o^i_t and a^i_t represent the observation and action at time t in trajectory i
- r̂(o^i_t, a^i_t) is the estimated reward for that observation-action pair
- The formula uses the softmax function (exponential normalization):

P̂[σ¹ ≻ σ²] = exp(∑r̂(o¹_t, a¹_t)) / [exp(∑r̂(o¹_t, a¹_t)) + exp(∑r̂(o²_t, a²_t))]

This means:
1. Sum up all the predicted rewards along trajectory 1
2. Sum up all the predicted rewards along trajectory 2
3. Apply exponential function to both sums
4. The probability of preferring trajectory 1 is the ratio of exp(sum1) to the total exp(sum1) + exp(sum2)

3. The Loss Function

The goal is to find parameters for r̂ that make its predictions match the actual human preferences:

loss(r̂) = -∑ [μ(1)log P̂[σ¹ ≻ σ²] + μ(2)log P̂[σ² ≻ σ¹]]

Where:
- (σ¹, σ², μ) ∈ D means we're summing over all the comparison data in our dataset D
- μ is a distribution over {1,2} indicating which segment the human preferred
- If the human strictly preferred segment 1, then μ(1) = 1 and μ(2) = 0
- If the human strictly preferred segment 2, then μ(1) = 0 and μ(2) = 1
- If the human found them equal, then μ(1) = μ(2) = 0.5

This is the standard cross-entropy loss function used in classification problems, measuring how well our predicted probabilities match the actual human judgments.

4. The Bradley-Terry Model Connection

This approach is based on the Bradley-Terry model, which is a statistical model for paired comparisons. It's similar to:

1. The Elo rating system in chess: Players have ratings, and the difference in ratings predicts the probability of one player beating another.

2. In this case: Trajectory segments have "ratings" (the sum of rewards), and the difference in ratings predicts the probability of a human preferring one segment over another.

In essence, the reward function learns to assign higher values to states and actions that humans tend to prefer, creating a preference scale that can be used to guide the agent's behavior.
"""


The most important idea that we need to take forth from this paper is.

We can use RLHF from non-expert humans for a fraction of cost by comparing stuff.

Fun story: One time researchers tried to RL a helicopter and it started flying backwards

### PPO: Proximal Policy Optimization

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

Read the following blogs to understand these topics better then explain them 
- https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
- http://www.scholarpedia.org/article/Policy_gradient_methods
- https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
- https://jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146
- https://jonathan-hui.medium.com/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9
- http://www.incompleteideas.net/book/first/ebook/node43.html
- https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo
- https://huggingface.co/blog/NormalUhr/rlhf-pipeline
- https://iclr-blogposts.github.io/2024/blog/the-n-implementation-details-of-rlhf-with-ppo/

- NOTE Explain the maths using baskets and fruits

Another big LLM algo that came out in 2017, and too again by OpenAI. Really goes to show how much they tried to advance AI and be public about it(Atleast in the early days). 

This is going to be math heavy so be prepared (Dw, I will guide you in each step)

"""
<details>
<summary>Quick Summary</summary>
I'll be your guide through this machine learning research paper, focusing on building intuition for the mathematical concepts while waiting for your specific questions to dive deeper.

## High-Level Summary: Proximal Policy Optimization (PPO)

This paper by John Schulman et al. from OpenAI introduces Proximal Policy Optimization (PPO), a family of policy gradient methods for reinforcement learning that achieves the reliability and data efficiency of Trust Region Policy Optimization (TRPO) while being much simpler to implement and more compatible with various neural network architectures.

Key contributions:
- A novel "clipped" surrogate objective function that provides a pessimistic estimate of policy performance
- An algorithm that alternates between data collection and multiple epochs of optimization on the same data
- Empirical validation showing PPO outperforms other online policy gradient methods across continuous control tasks and Atari games
- A balance between sample complexity, implementation simplicity, and computation time

The core innovation is their clipped probability ratio approach, which constrains policy updates without requiring the complex second-order optimization techniques used in TRPO. This makes PPO more practical while maintaining performance guarantees.

Feel free to ask specific questions about any aspect of the paper, and I'm happy to explore the mathematical formulations, algorithm details, or empirical results in greater depth.
</details>
"""

**Problem** """
However, there is room for improvement in developing a method that is scalable (to
large models and parallel implementations), data efficient, and robust (i.e., successful on a variety
of problems without hyperparameter tuning). Q-learning (with function approximation) fails on
many simple problems1 and is poorly understood, vanilla policy gradient methods have poor data
effiency and robustness; and trust region policy optimization (TRPO) is relatively complicated,
and is not compatible with architectures that include noise (such as dropout) or parameter sharing
(between the policy and value function, or with auxiliary tasks).
"""

**Solution** """
This paper seeks to improve the current state of affairs by introducing an algorithm that attains
the data efficiency and reliable performance of TRPO, while using only first-order optimization.
We propose a novel objective with clipped probability ratios, which forms a pessimistic estimate
(i.e., lower bound) of the performance of the policy. To optimize policies, we alternate between
sampling data from the policy and performing several epochs of optimization on the sampled **data**
"""

Policy Gradient Methods

"""
# Understanding Policy Gradient Methods

Let me break down this section on policy gradient methods step by step:

## Step 1: The Basic Idea
Policy gradient methods are a family of reinforcement learning algorithms that directly optimize a policy function by adjusting its parameters in the direction of greater expected rewards. They work by:
1. Collecting experience (state-action pairs and rewards) using the current policy
2. Estimating the policy gradient (the direction that would improve the policy)
3. Updating the policy parameters using this gradient

## Step 2: The Gradient Estimator
The core of policy gradient methods is the gradient estimator shown in equation (1). This formula tells us how to estimate the direction in which we should adjust our policy parameters to increase expected rewards:

The gradient estimator ĝ is an empirical average of the product of two terms:
- `∇θ log πθ(at|st)`: The gradient of the log probability of taking action at in state st
- `Ât`: An estimate of the advantage function, which tells us how much better action at is compared to the average action in state st

## Step 3: The Objective Function
Equation (2) shows the policy gradient objective function LPG(θ). In practice, modern implementations use automatic differentiation to compute the gradient. They set up an objective function whose gradient is the policy gradient estimator, then let the automatic differentiation calculate the gradient.

## Step 4: The Problem with Multiple Optimization Steps
The authors point out an important issue: while it seems like a good idea to perform multiple optimization steps on the same batch of data (to get more out of each data collection), this approach often leads to destructively large policy updates. This happens because:
1. The data was collected using an older version of the policy
2. As the policy changes during optimization, the data becomes less representative of the current policy's behavior
3. This mismatch can lead to overconfident and harmful updates

This observation motivates the need for the "proximal" part of PPO, which constrains how much the policy can change during these multiple optimization steps.

<details>
<summary>Mathematical Notation</summary>

- $\hat{g}$: Estimator of the policy gradient
- $\mathbb{Ê}_t$: Empirical average over a finite batch of samples
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$: Gradient of the log probability of taking action $a_t$ in state $s_t$ under policy $\pi_\theta$
- $\hat{A}_t$: Estimator of the advantage function at timestep $t$
- $\pi_\theta$: A stochastic policy parameterized by $\theta$
- $s_t$: State at timestep $t$
- $a_t$: Action at timestep $t$
- $L^{PG}(\theta)$: Policy gradient objective function
- $\log$: Natural logarithm
- $\theta$: Policy parameters
</details>

"""

"""
# Detailed Mathematical Analysis of Policy Gradient Methods

Let me focus specifically on the mathematical details of the policy gradient formulation:

## The Policy Gradient Estimator (Equation 1)

$$\hat{g} = \mathbb{\hat{E}}_t\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\hat{A}_t\right]$$

Breaking this down mathematically:

1. **Policy Parameterization**: $\pi_\theta(a_t|s_t)$ is a probability distribution over actions conditioned on the state, parameterized by $\theta$ (typically neural network weights). For each state $s_t$, it outputs a probability for each possible action $a_t$.

2. **Log-Probability Gradient**: $\nabla_\theta \log \pi_\theta(a_t|s_t)$ computes the gradient of the log probability with respect to policy parameters $\theta$. This is a vector pointing in the direction that would increase the probability of taking action $a_t$ in state $s_t$.

   - If $\pi_\theta$ is a Gaussian policy for continuous actions with mean $\mu_\theta(s_t)$ and standard deviation $\sigma_\theta(s_t)$, then:
     $$\log \pi_\theta(a_t|s_t) = -\frac{(a_t - \mu_\theta(s_t))^2}{2\sigma_\theta(s_t)^2} - \log(\sigma_\theta(s_t)) - \frac{1}{2}\log(2\pi)$$
     
   - If $\pi_\theta$ is a categorical policy for discrete actions with probabilities $p_\theta(a|s_t)$ for each action $a$, then:
     $$\log \pi_\theta(a_t|s_t) = \log(p_\theta(a_t|s_t))$$

3. **Advantage Estimation**: $\hat{A}_t$ is an estimator of the advantage function, which represents how much better action $a_t$ is compared to the average action in state $s_t$. Mathematically:
   $$\hat{A}_t \approx Q(s_t, a_t) - V(s_t)$$
   
   Where $Q(s_t, a_t)$ is the action-value function (expected return of taking action $a_t$ in state $s_t$) and $V(s_t)$ is the state-value function (expected return from state $s_t$).

4. **Empirical Expectation**: $\mathbb{\hat{E}}_t[\cdot]$ indicates an empirical average over a batch of collected samples:
   $$\mathbb{\hat{E}}_t[f(t)] = \frac{1}{T}\sum_{t=1}^T f(t)$$
   
   Where $T$ is the number of timesteps in the collected batch.

## The Policy Gradient Objective (Equation 2)

$$L^{PG}(\theta) = \mathbb{\hat{E}}_t\left[\log \pi_\theta(a_t|s_t)\hat{A}_t\right]$$

This objective function is constructed so that its gradient with respect to $\theta$ equals the policy gradient estimator in Equation 1:

$$\nabla_\theta L^{PG}(\theta) = \mathbb{\hat{E}}_t\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\hat{A}_t\right] = \hat{g}$$

The mathematical derivation works because:
1. The advantage estimator $\hat{A}_t$ doesn't depend on $\theta$ (it's treated as a constant when differentiating)
2. Therefore: $\nabla_\theta(\log \pi_\theta(a_t|s_t)\hat{A}_t) = \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t$
3. The expectation operator and gradient operator can be exchanged (under mild conditions)

## Implementation Detail

When implementing this in practice with automatic differentiation frameworks (like TensorFlow or PyTorch), we:
1. Compute $\hat{A}_t$ values for our collected batch of data
2. Set up the objective function $L^{PG}(\theta)$ 
3. Let the automatic differentiation compute the gradient
4. Apply this gradient using a stochastic gradient ascent algorithm:
   $$\theta_{new} = \theta_{old} + \alpha \cdot \hat{g}$$
   Where $\alpha$ is the learning rate

```
<details>
<summary>Mathematical Notation</summary>

- $\hat{g}$: Estimator of the policy gradient
- $\mathbb{\hat{E}}_t[\cdot]$: Empirical average over a finite batch of samples ($\frac{1}{T}\sum_{t=1}^T$)
- $\nabla_\theta$: Gradient operator with respect to parameters $\theta$
- $\log \pi_\theta(a_t|s_t)$: Log probability of taking action $a_t$ in state $s_t$ under policy $\pi_\theta$
- $\hat{A}_t$: Estimator of the advantage function at timestep $t$
- $\pi_\theta$: A stochastic policy parameterized by $\theta$
- $s_t$: State at timestep $t$
- $a_t$: Action at timestep $t$
- $L^{PG}(\theta)$: Policy gradient objective function
- $Q(s_t, a_t)$: Action-value function
- $V(s_t)$: State-value function
- $\theta$: Policy parameters
- $\alpha$: Learning rate for gradient ascent
</details>
```
"""

TRPO 

"""
# Detailed Mathematical Analysis of Trust Region Methods

Let me focus specifically on the mathematical formulation of Trust Region Policy Optimization (TRPO):

## The TRPO Objective (Equations 3-4)

The TRPO algorithm formulates policy optimization as a constrained optimization problem:

$$
\begin{align}
\text{maximize}_\theta \quad & \mathbb{\hat{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t\right] \quad \quad (3)\\
\text{subject to} \quad & \mathbb{\hat{E}}_t[\text{KL}[\pi_{\theta_{old}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]] \leq \delta \quad (4)
\end{align}
$$

Breaking this down mathematically:

1. **Probability Ratio**: The term $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the ratio of probabilities under the new policy $\pi_\theta$ versus the old policy $\pi_{\theta_{old}}$. This ratio:
   - Equals 1 when the policies assign equal probability to the action
   - Is greater than 1 when the new policy makes the action more likely
   - Is less than 1 when the new policy makes the action less likely

2. **Surrogate Objective**: The objective $\mathbb{\hat{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t\right]$ can be understood as:
   - When $\hat{A}_t > 0$ (good actions), increase their probability (maximize ratio)
   - When $\hat{A}_t < 0$ (bad actions), decrease their probability (minimize ratio)

3. **KL Divergence Constraint**: The term $\text{KL}[\pi_{\theta_{old}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]$ measures the difference between the old and new policy distributions. For two distributions $P$ and $Q$, the KL divergence is defined as:
   $$\text{KL}[P||Q] = \mathbb{E}_{x \sim P}\left[\log\frac{P(x)}{Q(x)}\right]$$
   
   - For continuous action spaces with Gaussian policies:
     $$\text{KL}[\mathcal{N}(\mu_1,\Sigma_1)||\mathcal{N}(\mu_2,\Sigma_2)] = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} + \text{Tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1) - d\right]$$
     where $d$ is the dimension of the action space.
   
   - For discrete action spaces:
     $$\text{KL}[P||Q] = \sum_{a} P(a) \log\frac{P(a)}{Q(a)}$$

4. **Constraint Parameter $\delta$**: This hyperparameter controls how much the policy is allowed to change in a single update. Typically small values (e.g., 0.01-0.05) are used.

## The Penalized Version (Equation 5)

Instead of using a hard constraint, the theory also supports a penalized objective:

$$\text{maximize}_\theta \mathbb{\hat{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t - \beta \text{KL}[\pi_{\theta_{old}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]\right] \quad (5)$$

This reformulates the constrained optimization as an unconstrained one:

1. **Penalty Coefficient $\beta$**: This parameter balances between maximizing the surrogate objective and minimizing the KL divergence.
   - Large $\beta$: Conservative updates that change the policy very little
   - Small $\beta$: Aggressive updates that may significantly change the policy

2. **Mathematical Equivalence**: Under certain conditions, for any constraint $\delta$ in equation (4), there exists a $\beta$ in equation (5) that gives the same solution.

3. **Practical Challenge**: The paper notes that finding a single value of $\beta$ that works well across different problems (or even at different stages of the same problem) is difficult, which is why TRPO uses the constrained formulation instead.

4. **Mathematical Insight**: The paper mentions that a surrogate objective using the max KL (rather than mean KL) forms a lower bound (pessimistic estimate) on policy performance. This theoretical foundation justifies the constraint-based approach.

## Implementation Details

When solving this constrained optimization problem:

1. TRPO applies a linear approximation to the objective:
   $$\mathbb{\hat{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t\right] \approx \mathbb{\hat{E}}_t\left[\hat{A}_t\right] + \mathbb{\hat{E}}_t\left[\frac{\partial}{\partial \theta}\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\bigg|_{\theta=\theta_{old}}\right]^T(\theta - \theta_{old})$$

2. And a quadratic approximation to the constraint:
   $$\mathbb{\hat{E}}_t[\text{KL}[\pi_{\theta_{old}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]] \approx \frac{1}{2}(\theta - \theta_{old})^T H (\theta - \theta_{old})$$
   where $H$ is the Hessian of the KL divergence with respect to $\theta$.

3. The conjugate gradient algorithm is then used to efficiently solve this approximated problem.

```
<details>
<summary>Mathematical Notation</summary>

- $\pi_\theta(a_t|s_t)$: Probability of taking action $a_t$ in state $s_t$ under policy with parameters $\theta$
- $\pi_{\theta_{old}}(a_t|s_t)$: Probability under previous policy parameters
- $\theta_{old}$: Vector of policy parameters before the update
- $\hat{A}_t$: Advantage function estimator at timestep $t$
- $\mathbb{\hat{E}}_t[\cdot]$: Empirical average over collected samples
- $\text{KL}[P||Q]$: Kullback-Leibler divergence between distributions $P$ and $Q$
- $\delta$: Constraint parameter limiting the size of policy update
- $\beta$: Coefficient for KL penalty in the unconstrained formulation
- $\pi_{\theta_{old}}(\cdot|s_t)$: Complete action distribution under old policy for state $s_t$
- $\pi_\theta(\cdot|s_t)$: Complete action distribution under new policy for state $s_t$
- $H$: Hessian matrix of the KL divergence with respect to policy parameters
</details>
```
"""

Clipped Surogate Objective 

"""
# Detailed Mathematical Analysis of Clipped Surrogate Objective

Let me focus on the mathematical details of the Clipped Surrogate Objective, which is the core innovation of PPO:

## The Probability Ratio and CPI Objective

First, the paper defines a probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

This ratio measures how the probability of taking action $a_t$ in state $s_t$ changes under the new policy $\pi_\theta$ compared to the old policy $\pi_{\theta_{old}}$. An important property is that $r(\theta_{old}) = 1$, since at $\theta = \theta_{old}$, the policies are identical.

The Conservative Policy Iteration (CPI) objective from previous work is defined as:

$$L^{CPI}(\theta) = \mathbb{\hat{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t\right] = \mathbb{\hat{E}}_t\left[r_t(\theta)\hat{A}_t\right] \quad (6)$$

This objective encourages:
- Increasing probability (making $r_t(\theta) > 1$) for actions with positive advantage ($\hat{A}_t > 0$)
- Decreasing probability (making $r_t(\theta) < 1$) for actions with negative advantage ($\hat{A}_t < 0$)

## The Clipped Surrogate Objective (Equation 7)

The key innovation of PPO is the clipped surrogate objective:

$$L^{CLIP}(\theta) = \mathbb{\hat{E}}_t\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right] \quad (7)$$

Breaking this down mathematically:

1. **Clipping Function**: The function $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$ constrains the probability ratio to the interval $[1-\epsilon, 1+\epsilon]$:
   $$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) = \begin{cases}
   1-\epsilon & \text{if } r_t(\theta) < 1-\epsilon \\
   r_t(\theta) & \text{if } 1-\epsilon \leq r_t(\theta) \leq 1+\epsilon \\
   1+\epsilon & \text{if } r_t(\theta) > 1+\epsilon
   \end{cases}$$

2. **The Minimum Operation**: The $\min$ operation in the objective creates a pessimistic estimate by taking the lower of two values:
   - The original surrogate objective $r_t(\theta)\hat{A}_t$
   - The clipped surrogate objective $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t$

3. **Case Analysis Based on Advantage Sign**:
   
   When $\hat{A}_t > 0$ (positive advantage):
   - The original objective $r_t(\theta)\hat{A}_t$ increases monotonically with $r_t(\theta)$
   - The clipped objective is constant at $(1+\epsilon)\hat{A}_t$ when $r_t(\theta) > 1+\epsilon$
   - Taking the minimum means we optimize $r_t(\theta)\hat{A}_t$ only when $r_t(\theta) \leq 1+\epsilon$
   - Beyond that, there's no incentive to increase $r_t(\theta)$ further
   
   When $\hat{A}_t < 0$ (negative advantage):
   - The original objective $r_t(\theta)\hat{A}_t$ decreases monotonically with $r_t(\theta)$
   - The clipped objective is constant at $(1-\epsilon)\hat{A}_t$ when $r_t(\theta) < 1-\epsilon$
   - Taking the minimum means we optimize $r_t(\theta)\hat{A}_t$ only when $r_t(\theta) \geq 1-\epsilon$
   - Below that, there's no incentive to decrease $r_t(\theta)$ further

4. **Mathematical Properties**:
   - First-order equivalence: $L^{CLIP}(\theta) = L^{CPI}(\theta)$ to first order around $\theta_{old}$ (i.e., at $r = 1$)
   - Explicit constraint: Unlike TRPO, the constraint is built directly into the objective function
   - Lower bound: $L^{CLIP}$ serves as a pessimistic bound (lower bound) on $L^{CPI}$
   - Automatic penalty: The clipping automatically penalizes large policy changes without requiring complex constrained optimization

5. **Hyperparameter $\epsilon$**: Controls the clipping threshold, typically set to a small value like 0.2. This determines how much the policy is allowed to change in a single update.

## Graphical Interpretation (Figure 1)

The paper includes graphs showing the behavior of a single term in $L^{CLIP}$ as a function of the probability ratio $r$:

1. For positive advantages ($\hat{A}_t > 0$):
   - The function increases linearly with $r$ until $r = 1+\epsilon$
   - After that, it plateaus, removing any incentive to increase $r$ beyond $1+\epsilon$
   
2. For negative advantages ($\hat{A}_t < 0$):
   - The function decreases linearly with $r$ until $r = 1-\epsilon$
   - Below that, it plateaus, removing any incentive to decrease $r$ below $1-\epsilon$

The red circle at $r = 1$ in both plots represents the starting point of optimization (the previous policy).

## Lower Bound Property (Figure 2)

Figure 2 (mentioned in the text) shows that $L^{CLIP}$ forms a lower bound on $L^{CPI}$, effectively penalizing large policy updates. This approximates the trust region method of TRPO but uses only first-order optimization.


<details>
<summary>Mathematical Notation</summary>

- $r_t(\theta)$: Probability ratio $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ at timestep $t$
- $\pi_\theta(a_t|s_t)$: Probability of action $a_t$ in state $s_t$ under policy with parameters $\theta$
- $\pi_{\theta_{old}}(a_t|s_t)$: Same probability under previous policy parameters
- $\theta_{old}$: Policy parameters before the update
- $\hat{A}_t$: Advantage function estimator at timestep $t$
- $\mathbb{\hat{E}}_t[\cdot]$: Empirical average over collected samples
- $L^{CPI}(\theta)$: Conservative Policy Iteration objective
- $L^{CLIP}(\theta)$: Clipped surrogate objective
- $\text{clip}(x,a,b)$: Function that clips $x$ to be within the interval $[a,b]$
- $\epsilon$: Clipping hyperparameter (typically 0.2)
- $\min(a,b)$: Function returning the minimum of $a$ and $b$
</details>
"""

### MOE : Mixture Of Experts 

[This came out in JAn so it should technically be at the top]

Another explosive paper, In 2017. Talk about being a crazy year right. Well to be perfectly honest MOE was actually introduced in 1990 in this [paper]()

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)


blogs 
- https://huggingface.co/blog/moe
- https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts

<details>
<summary>
Quick Summary
</summary>
# Brief Summary of "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"

This 2017 paper by Shazeer et al. introduces a novel approach to dramatically increase neural network capacity without proportionally increasing computational costs. The core innovation is the Sparsely-Gated Mixture-of-Experts (MoE) layer, which contains thousands of feed-forward neural networks (experts), with a trainable gating network that selectively activates only a small subset of experts for each input example.

Key highlights:
- The authors achieve over 1000x improvements in model capacity while maintaining computational efficiency
- Their approach addresses several challenges of conditional computation, including GPU utilization and load balancing
- When applied to language modeling and machine translation tasks, their MoE models significantly outperform state-of-the-art models with lower computational cost
- Their largest model contains up to 137 billion parameters and demonstrates continued performance improvements with increased capacity

This paper represents a significant advancement in scaling neural networks efficiently, presaging some of the techniques that would later become important in very large language models.

Is there a specific aspect of this paper you'd like to explore further?
</details>

**Problem** 
"""
The capacity of a neural network to absorb information is limited by its number of
parameters.
"""

**Solution**
"""
Conditional computation, where parts of the network are active on a
per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation.
"""

[Visualize the solution as taking a bunch of students, then training each to be really good at one topic. Add a disclaimer that this is just for intuition. In reality it has been observed that MoE models focus more on tokens rather than man-made concepts]

Understanding the Gating Network

[completely understand 2.1 and write that down]

ADDRESSING PERFORMANCE CHALLENGES [TALK_ABOUT_THESE_AS_WELL]

Load balancing loss,understand and explain that too

We are deviating a bit from what the paper proposed and moving into the future on how MoE is actually used. 
## 2018: BERT and Early Innovations

### Universal Language Model Fine-tuning for Text Classification

[paper](https://arxiv.org/pdf/1801.06146)

<details>
<summary>
Quick Summary
</summary>
"""
# Summary of "Universal Language Model Fine-tuning for Text Classification"

This 2018 paper by Jeremy Howard and Sebastian Ruder introduces ULMFiT (Universal Language Model Fine-tuning), a method for transfer learning in NLP tasks. The authors present an approach that mirrors the success of transfer learning in computer vision by using a pre-trained language model and fine-tuning it for specific text classification tasks.

Key contributions:
1. A three-stage approach to transfer learning for NLP tasks:
   - General-domain language model pretraining
   - Target task language model fine-tuning
   - Target task classifier fine-tuning

2. Novel fine-tuning techniques to prevent catastrophic forgetting:
   - Discriminative fine-tuning (using different learning rates for different layers)
   - Slanted triangular learning rates (a specific learning rate schedule)
   - Gradual unfreezing (progressively unfreezing layers from last to first)

3. State-of-the-art results on six text classification datasets with significant error reductions (18-24%)

4. Impressive sample efficiency - with just 100 labeled examples, the method matches the performance of training from scratch with 10-100x more data

The paper demonstrates that effective transfer learning is possible in NLP without task-specific modifications or architecture changes, using a standard 3-layer LSTM model with careful fine-tuning techniques.

Is there a particular aspect of the paper you'd like me to explain in more detail?
"""
"""
Inductive transfer learning has greatly impacted computer vision, but existing approaches in NLP still require task-specific
modifications and training from scratch.
We propose Universal Language Model
Fine-tuning (ULMFiT), an effective transfer learning method that can be applied to
any task in NLP, and introduce techniques
that are key for fine-tuning a language
model.
"""

</details>

#### ElMO: Embeddings from Language Models

[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

https://pythonandml.github.io/dlbook/content/word_embeddings/elmo.html

<details>
<summary>
Quick Summary
</summary>
# Brief Summary of "Deep contextualized word representations"

This paper introduces ELMo (Embeddings from Language Models), a new approach to creating word representations that capture both complex word characteristics (syntax and semantics) and how those characteristics change across different contexts (addressing polysemy). Unlike traditional word embeddings that assign a single vector per word, ELMo derives representations from a bidirectional language model (biLM) pre-trained on a large text corpus.

The key innovation is that ELMo representations are deep - they're a function of all internal layers of the biLM, not just the final layer. The authors show that different layers capture different linguistic properties (lower layers capture syntax, higher layers capture semantics). By learning task-specific weightings of these layers, models can access both types of information simultaneously.

The authors demonstrate that adding ELMo to existing models significantly improves performance across six diverse NLP tasks, including question answering, textual entailment, sentiment analysis, and named entity recognition - achieving state-of-the-art results in all cases, with relative error reductions ranging from 6-20%.

Is there a specific aspect of this paper you'd like me to elaborate on?
</details>

**Problem**

learning high quality representations can be challenging. They should ideally
model both (1) complex characteristics of word
use (e.g., syntax and semantics), and (2) how these
uses vary across linguistic contexts (i.e., to model
polysemy).

**Solution**

"""
Our representations differ from traditional word
type embeddings in that each token is assigned a
representation that is a function of the entire input
sentence. We use vectors derived from a bidirectional LSTM that is trained with a coupled language model (LM) objective on a large text corpus.
"""


### GPT-1

[paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
[blog](https://openai.com/index/language-unsupervised/)

This was the beginning of the era we live in now

- Unidirectional decoder
- BPE tokenization
- Zero-shot capabilities
- Language modeling objective

"""
Link: https://huggingface.co/docs/transformers/model_doc/openai-gpt
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension:
Application: Text generation, but adaptable to many other NLP tasks when fine tuned.
Date (of first known publication): 06/2018
Num. Params:117M
Corpus: Unsupervised Pretraining on BookCorpus dataset. Supervised Finetuning on several task-specific datasets including SNLI, RACE, Quora…
License: N/A
Lab: OpenAI
"""

"""
Our system works in two stages; first we train a transformer model on a very large amount of data in an unsupervised manner—using language modeling as a training signal—then we fine-tune this model on much smaller supervised datasets to help it solve specific tasks.
"""

Training a GPT

Semi-supervised Sequence Learning

<details>
<summary>
Quick Summary
</summary>

# Improving Language Understanding by Generative Pre-Training: A Brief Summary

This seminal 2018 paper from OpenAI researchers (Radford, Narasimhan, Salimans, and Sutskever) introduces a powerful semi-supervised approach to natural language understanding that combines unsupervised pre-training with supervised fine-tuning.

The key innovation lies in training a large transformer-based language model on unlabeled text data, then leveraging the learned representations by fine-tuning this model on specific downstream tasks. This approach addresses a fundamental challenge in NLP: the scarcity of labeled data for various language understanding tasks.

The authors demonstrate that their method significantly outperforms task-specific architectures across 9 out of 12 NLP tasks, including natural language inference, question answering, semantic similarity, and text classification. Notable improvements include:
- 8.9% on commonsense reasoning (Stories Cloze Test)
- 5.7% on question answering (RACE)
- 1.5% on textual entailment (MultiNLI)

This approach minimizes task-specific architecture modifications by using "task-aware input transformations," which convert structured inputs into a sequence format compatible with the pre-trained model.

This paper laid important groundwork for later transformer-based language models, demonstrating that generative pre-training on unlabeled data could significantly improve performance on downstream language understanding tasks.

What aspects of this paper would you like me to explore in more detail?
</details>

**Problem**

"""
The ability to learn effectively from raw text is crucial to alleviating the dependence on supervised
learning in natural language processing (NLP). Most deep learning methods require substantial
amounts of manually labeled data, which restricts their applicability in many domains that suffer
from a dearth of annotated resources 
"""

**Solution**

"""

"""

blog - https://towardsdatascience.com/understanding-the-evolution-of-gpt-part-1-an-in-depth-look-at-gpt-1-and-what-inspired-it-b7388a32e87d/#:~:text=GPT%2D1%20is%20the%20first,standard%20procedure%20for%20NLP%20tasks.



Unsupervised pre-training

Supervised fine-tuning

Model Specification 

"""
Model specifications Our model largely follows the original transformer work [62]. We trained a
12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12
attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states.
We used the Adam optimization scheme [27] with a max learning rate of 2.5e-4. The learning rate
was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.
We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens.
Since layernorm [2] is used extensively throughout the model, a simple weight initialization of
N(0, 0.02) was sufficient. We used a bytepair encoding (BPE) vocabulary with 40,000 merges [53]
and residual, embedding, and attention dropouts with a rate of 0.1 for regularization. We also
employed a modified version of L2 regularization proposed in [37], with w = 0.01 on all non bias or
gain weights. For the activation function, we used the Gaussian Error Linear Unit (GELU) [18]. We
used learned position embeddings instead of the sinusoidal version proposed in the original work.
We use the ftfy library2
to clean the raw text in BooksCorpus, standardize some punctuation and
whitespace, and use the spaCy tokenizer.3

"""

### Sentencepiece

https://huggingface.co/docs/transformers/en/tokenizer_summary

https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15/

Wordpiece 

Unigram 

BPE https://arxiv.org/abs/1508.07909

[paper](https://arxiv.org/abs/1808.06226)

<details>
<summary>
Quick Summary
</summary>

"""
I'll be happy to serve as your guide through the machine learning research paper you've shared, adapting my explanations to your questions while helping you build stronger mathematical intuition.

# Brief Summary of "SentencePiece"

This paper introduces SentencePiece, an open-source subword tokenizer and detokenizer designed specifically for neural text processing, including Neural Machine Translation (NMT). The key innovation of SentencePiece is that it can train subword models directly from raw sentences without requiring pre-tokenization, enabling truly end-to-end and language-independent text processing.

The authors highlight several important features:

1. It implements two subword segmentation algorithms: byte-pair encoding (BPE) and unigram language model
2. It provides lossless tokenization that preserves all information needed to reconstruct the original text
3. The model is fully self-contained, ensuring reproducibility across implementations
4. It offers efficient training and segmentation algorithms
5. It includes library APIs for on-the-fly processing

They validate their approach through experiments on English-Japanese translation, showing comparable accuracy to systems that use pre-tokenization, while being significantly faster for non-segmented languages like Japanese.

I'm ready to discuss any specific aspects of the paper you'd like to explore in more detail.
"""
</details>

**Problem** Tough to make NMT language independent 

**Solution**

"""
SentencePiece comprises four main components:
Normalizer, Trainer, Encoder, and Decoder.
Normalizer is a module to normalize semanticallyequivalent Unicode characters into canonical
forms. Trainer trains the subword segmentation
model from the normalized corpus. We specify a
type of subword model as the parameter of Trainer.
Encoder internally executes Normalizer to normalize the input text and tokenizes it into a subword sequence with the subword model trained by
Trainer. Decoder converts the subword sequence
into the normalized tex
"""

#### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[paper](https://arxiv.org/abs/1810.04805)

https://jalammar.github.io/illustrated-bert/
https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
https://huggingface.co/blog/bert-101


- Bidirectional encoder
- WordPiece tokenization
- [CLS] and [SEP] tokens
- NSP and MLM objectives

"""
Link: https://huggingface.co/docs/transformers/model_doc/bert
Family: BERT
Pretraining Architecture: Encoder
Pretraining Task: MLM/NSP
Extension:It can be seen as a generalization of BERT and GPT in that it combines ideas from both in the encoder and decoder
Application:General Language Understanding and Question Answering. Many other language applications followed
Date (of first known publication): 10/2018
Num. Params:Base = 110M, Large = 340MT
Corpus:Toronto Book Corpus and Wikipedia (3.3B Tokens)
License: Open, Apache-2.0
Lab:Google
"""

<details>
<summary>
Quick Summary

I'll help you understand the BERT paper as requested. Let me provide a brief, high-level summary first.

## Brief Summary of "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"

This paper introduces BERT (Bidirectional Encoder Representations from Transformers), a groundbreaking language representation model that significantly advanced the state of natural language processing in 2018. The key innovation of BERT is its ability to pre-train deep bidirectional representations from unlabeled text, unlike previous models that were limited to unidirectional contexts (either left-to-right or right-to-left).

BERT employs two novel pre-training tasks:
1. **Masked Language Model (MLM)**: Randomly masks some percentage of input tokens and predicts those masked tokens
2. **Next Sentence Prediction (NSP)**: Predicts whether two sentences follow each other in original text

These pre-training objectives allow BERT to create context-aware representations that capture information from both left and right contexts. After pre-training on large text corpora (BookCorpus and Wikipedia), BERT can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks without task-specific architecture modifications.

The paper demonstrated significant improvements over previous methods on eleven NLP tasks, including the GLUE benchmark, SQuAD, and SWAG datasets.

Is there a specific aspect of BERT that you'd like me to explain in more detail?
</summary>
</details>

This paper wasn't trying to find a problem then solve it per say. It is more of an innovation 

"""
BERT is designed to pretrain deep bidirectional representations from
unlabeled text by jointly conditioning on both
left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer
to create state-of-the-art models for a wide
range of tasks, such as question answering and
language inference, without substan
"""



## 2019: Scaling and Efficiency

### GPT-2

https://jalammar.github.io/illustrated-gpt2/

[blog](https://openai.com/index/better-language-models/)
[paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

"""
Link: https://huggingface.co/docs/transformers/model_doc/gpt2
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: Minor extensions to the GPT architecture (e.g. layer normalization moved to the input of each sub-layer, or increased context size from 512 to 1024)
Application: Text generation, but adaptable to many other NLP tasks when fine tuned.
Date (of first known publication): 02/2019
Num. Params: 124M, 355M, 774M, 1.5B
Corpus: 8 million web pages (40 GB). 10X GPT . WebText dataset is created by crawling all links at Reddit with at least 3 Karma points.
License: Open, Modified MIT license
Lab: OpenAI
"""

<details>
<summary>
Quick Summary
</summary>

# Summary of "Language Models are Unsupervised Multitask Learners"

This 2019 paper by Radford et al. (OpenAI) introduces GPT-2, a large-scale language model that demonstrates impressive zero-shot learning capabilities across multiple NLP tasks. The key insight of this paper is that language models trained on sufficiently large and diverse datasets naturally acquire the ability to perform various language tasks without explicit supervision.

Key contributions:
1. Introduction of WebText - a high-quality web dataset created by scraping outbound links from Reddit with at least 3 karma
2. Development of GPT-2, a Transformer-based language model with 1.5 billion parameters
3. Demonstration that a single unsupervised language model can perform multiple NLP tasks without task-specific training
4. Evidence that model performance scales in a log-linear fashion with model size

The paper shows that GPT-2 achieves state-of-the-art results on 7 out of 8 tested language modeling datasets in a zero-shot setting. It also demonstrates promising zero-shot performance on tasks like reading comprehension, summarization, translation, and question answering without any task-specific fine-tuning.

This work represents a significant step toward building more general NLP systems that can learn to perform tasks from naturally occurring demonstrations in text, rather than requiring task-specific datasets and architectures for each application.
</details>
"""
Common Crawl. Trinh & Le (2018)’s best results were
achieved using a small subsample of Common Crawl which
included only documents most similar to their target dataset,
the Winograd Schema Challenge. While this is a pragmatic
approach to improve performance on a specific task, we
want to avoid making assumptions about the tasks to be
performed ahead of time.
Instead, we created a new web scrape which emphasizes
document quality. To do this we only scraped web pages
which have been curated/filtered by humans. Manually
filtering a full web scrape would be exceptionally expensive
so as a starting point, we scraped all outbound links from
Reddit, a social media platform, which received at least 3
karma. This can be thought of as a heuristic indicator for
whether other users found the link interesting, educational,
or just funny.
The resulting dataset, WebText, contains the text subset
of these 45 million links. To extract the text from HTML
responses we use a combination of the Dragnet (Peters &
Lecocq, 2013) and Newspaper1
content extractors. All results presented in this paper use a preliminary version of
WebText which does not include links created after Dec
2017 and which after de-duplication and some heuristic
based cleaning contains slightly over 8 million documents
for a total of 40 GB of text. We removed all Wikipedia
documents from WebText since it is a common data source
for other datasets and could complicate analysis due to over
"""


### RoBERTa



[paper](https://arxiv.org/abs/1907.11692)

- Dynamic masking
- Removed NSP
- Larger batch sizes
- Extended training

<details>
<summary>
Quick Summary
</summary>

# Summary of "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

This 2019 paper by Liu et al. from Facebook AI presents RoBERTa (Robustly Optimized BERT Pretraining Approach), which demonstrates that BERT was significantly undertrained and can achieve state-of-the-art performance with careful optimization choices.

Key contributions:

1. The paper identifies several critical design decisions that significantly improve BERT's performance:
   - Training the model longer with larger batches over more data
   - Removing the Next Sentence Prediction (NSP) objective
   - Training on longer sequences
   - Dynamically changing the masking pattern applied to training data

2. The researchers collect a larger dataset (including a new CC-NEWS corpus) to better control for training set size effects.

3. Through extensive experimentation, they show that when properly optimized, BERT's masked language modeling objective is competitive with newer approaches like XLNet.

4. RoBERTa achieves state-of-the-art results on GLUE, RACE, and SQuAD benchmarks without multi-task fine-tuning for GLUE or additional data for SQuAD.

The authors emphasize that seemingly mundane training decisions (like batch size, training time, and dataset size) can have as much impact on final performance as architectural innovations. This raises important questions about the source of improvements in recent NLP models and highlights the need for careful replication studies.

The paper is particularly notable for its thorough empirical analysis of training hyperparameters and careful ablation studies showing the contribution of each modification to overall performance.
</details>

"""
Link: https://huggingface.co/docs/transformers/model_doc/roberta
Family: BERT
Pretraining Architecture: Encoder
Pretraining Task: MLM (Dynamic)
Extension: Extension of BERT with optimized training procedure and more data
Application: Same as BERT
Date (of first known publication): 07/2019
Num. Params: 356M
Corpus: Same as BERT + CC News + OpenWebText + Stories ( 33B Tokens)
License: N/A
Lab: UW/Google
"""

### DistilBERT and Model Compression

[paper](https://arxiv.org/abs/1910.01108)

- Knowledge distillation techniques
- Parameter sharing
- Pruning strategies
- Quantization methods

"""
Link: https://huggingface.co/docs/transformers/model_doc/distilbert
Family: BERT
Pretraining Architecture: Encoder
Pretraining Task: MLM/NSP
Extension: Compressed version of BERT using distillation, which is much more efficient given the same number of parameters
Application: Same as BERT
Date (of first known publication): 10/2019
Num. Params:66M
Corpus: Same as BERT
License: Open, Apache-2.0
Lab: Huggingface
"""

### BART

[paper](https://arxiv.org/abs/1910.13461)

"""
Link: https://huggingface.co/docs/transformers/model_doc/bart
Family: BERT for encoder, GPT for Decoder
Pretraining Architecture: Encoder/Decoder
Pretraining Task: DAE
Extension: It can be seen as a generalization of BERT and GPT in that it combines ideas from both in the encoder and decoder
Application: Mostly text generation but also some text understanding tasks
Date (of first known publication): 10/2019
Num. Params: Base = 140M, Large = 400M. In general, roughly 10% larger than BART for equivalent architectures.
Corpus:Same as RoBERTa (160Gb of news, books, stories)
License: Open, Apache-2.0
Lab:Facebook
"""

### XLNet

[paper](https://arxiv.org/abs/1906.08237)

- Permutation-based training approach
- Surpassed BERT on multiple benchmarks

### Megatron

[paper](https://arxiv.org/abs/1909.08053)

- Model parallelism for efficient large model training

### Sparse Attention Patterns

[paper](https://arxiv.org/abs/1904.10509)

- Reduced computational complexity for long sequences

## 2020: The Scale Revolution

### Reformer: The Efficient Transformer

[paper](https://arxiv.org/abs/2001.04451)

### Longformer: The Long-Document Transformer

[paper](https://arxiv.org/abs/2004.05150)

### GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

[paper](https://arxiv.org/abs/2006.16668)

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

[paper](https://arxiv.org/abs/2005.11401)

### Big Bird: Transformers for Longer Sequences

[paper](https://arxiv.org/abs/2007.14062)

### GPT-3

[paper](https://arxiv.org/abs/2005.14165)

- In-context learning
- Few-shot capabilities
- Scaling laws discovery
- Batch size scaling

### Rethinking Attention with Performers

[paper](https://arxiv.org/abs/2009.14794v4)

### T5

[paper](https://arxiv.org/abs/1910.10683)

- Encoder-decoder architecture
- Unified text-to-text framework
- Span corruption
- Multi-task pre-training

"""
Link: https://huggingface.co/docs/transformers/model_doc/t5
Family: Transformer
Pretraining Architecture: Encoder/Decoder
Pretraining Task: DAE
Extension: Same as original Transformer with some additions such as relative positional embeddings like Transformer XL
Application: General language tasks including machine translation, question answering, abstractive summarization, and text classification
Date (of first known publication): 10/2019
Num. Params: 11 B (up to)
Corpus: Colossal Clean Crawled Corpus (C4) — Cleaned up version of the Common Crawl dataset — 750 GB
License: Open, Apache-2.0
Lab: Google
"""

### Measuring Massive Multitask Language Understanding

[paper](https://arxiv.org/abs/2009.03300)

### ZeRO (Zero Redundancy Optimizer)

[paper](https://arxiv.org/abs/1910.02054)

- Memory optimization for distributed training

### ELECTRA

[paper](https://arxiv.org/abs/2003.10555)

Google's model that used a discriminative approach instead of masked language modeling, providing more efficient training As noted, "Electra deploys a 'Masked Language Modeling' approach that masks certain words and trains the model to predict them. Additionally, Electra incorporates a 'Discriminator' network that aids in comprehending language without the need to memorize the training data."

### Switch Transformer

[paper](https://arxiv.org/abs/2101.03961)

Google's early mixture-of-experts approach that demonstrated trillion-parameter scale was possible

### Scaling Laws

[paper](https://arxiv.org/abs/2001.08361)

OpenAI's publication on the mathematical relationships between model size, dataset size, and computational budget demonstrated predictable patterns for improving performance This was part of the GPT-3 research which showed "that scaling up language models greatly improves task-agnostic, few-shot performance."

## 2021: Instruction Tuning and Alignment

### RoFormer: Enhanced Transformer with Rotary Position Embedding

[paper](https://arxiv.org/abs/2104.09864)

### Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

[paper](https://arxiv.org/abs/2104.04473)

### Transcending Scaling Laws with 0.1% Extra Compute

[paper](https://arxiv.org/abs/2210.11399)

### Improving language models by retrieving from trillions of tokens

[paper](https://arxiv.org/abs/2112.04426)

### CLIP

https://openai.com/index/clip/
Briefly talk about

### Dall-e

Briefly talk about

### FSDP

[paper](https://arxiv.org/abs/2304.11277)

### HumanEval

[paper](Evaluating Large Language Models Trained on Code)

### LoRA

[paper](https://arxiv.org/abs/2106.09685)

### Self-Instruct: Aligning Language Models with Self-Generated Instructions

[paper](https://arxiv.org/abs/2212.10560)

### PaLM

[paper](PaLM: Scaling Language Modeling with Pathways)

- Pathways system
- Scaled dot product attention
- Multi-query attention
- Parallel training techniques

"""
Link: https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html
Family: Transformer
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: Palm uses a typical decoder-only Transformer architecture, but adds quite a few extensions: SwiGLU activations, parallel layers, multi-query attention, RoPE embeddings, Shared Input-Output Embeddings, no biases, and a 256k SentencePiece vocabulary generated from the training data.
Application: PalM is designed as a general purpose language model with applicability to hundreds of different language tasks
Date (of first known publication): 04/2022
Num. Params: 540B
Corpus: 780B tokens from filtered webpages, books, Wikipedia, news articles, source code, and social media conversations. Code includes 24 programming languages.
License: Closed source, Accessible through API
Lab: Google
"""

### Gopher (DeepMind)

[paper](https://arxiv.org/abs/2112.11446)

- 280B parameter model released in December 2021 DeepMind introduced this model as a "280 billion parameter model" that was "evaluated on 152 diverse tasks, achieving state-of-the-art performance across the majority."
- Demonstrated significant scaling benefits in reading comprehension and fact-checking
- Represented a major advancement in model scale from DeepMind

### Megatron-Turing NLG

[paper](https://arxiv.org/abs/2201.11990)

- 530B parameter model announced in October 2021
- Combined Microsoft's Turing and NVIDIA's Megatron technologies
- Demonstrated advanced distributed training techniques
- Applied significant hardware optimization for large-scale training

## 2022: Democratization

### EFFICIENTLY SCALING TRANSFORMER INFERENCE

[paper](https://arxiv.org/pdf/2211.05102)

### Fast Inference from Transformers via Speculative Decoding

[paper](https://arxiv.org/abs/2211.17192)

### Chinchilla

[paper](https://arxiv.org/abs/2203.15556)
While the Chinchilla model itself wasn't released until 2022, the research behind it began in 2021, establishing important scaling principles that:

- Showed optimal token-to-parameter ratios should be approximately 20:1 This research found that "we need around 20 text tokens per parameter" for optimal training.
- Demonstrated many existing models were significantly undertrained
- Influenced the training methodology of subsequent models

"""
Link: https://arxiv.org/abs/2203.15556
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: Same as Gopher but with optimizations to reduce model size and therefore training/inference time with equal or superior performance
Application: Same as Gopher/GPT3
Date (of first known publication): 03/2022
Num. Params:70B
Corpus: Massive Text
License: Closed source.
Lab: Deepmind
"""

### Chain-of-thought prompting

[paper](https://arxiv.org/abs/2201.11903)

### InstructGPT

[paper](https://arxiv.org/abs/2203.02155)

https://openai.com/index/instruction-following/

- RLHF pipeline [blog on the topic](https://huggingface.co/blog/rlhf) & [blog 2](https://wandb.ai/ayush-thakur/RLHF/reports/Understanding-Reinforcement-Learning-from-Human-Feedback-RLHF-Part-1--VmlldzoyODk5MTIx)
- PPO implementation
- Human feedback collection
- Alignment techniques

"""
Link: https://github.com/openai/following-instructions-human-feedback
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: GPTInstruct starts off with a pretrained GPT3 model and adds reward modeling through reinforcement learning after a supervised finetuning
Application: Knowledge-intensive dialog or language tasks
Date (of first known publication): 01/2022
Num. Params: Same as GPT3
Corpus: Same as GPT3 for pretraining, but finetuned and optimized using labeler data and prompts
License: Closed source, Accessible through API
Lab: OpenAI
"""

### BLOOM

[paper](https://arxiv.org/abs/2211.05100)

- Multilingual pre-training
- Carbon footprint considerations
- Distributed training
- Community governance

"""
Link: https://huggingface.co/docs/transformers/model_doc/bloom
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: Main difference to GPT-3 is that it uses full attention instead of sparse attention
Application: Same as GPT-3
Date (of first known publication): 07/2022
Num. Params:176B
Corpus: 366B tokens (1.5 TB of text data) multilingual dataset
Lab: Big Science/Huggingface
License: Open, but need to follow restrictions in Attachment A, BigScience RAIL License v1.0
"""

### Emergent Abilities of Large Language Models

[paper](https://arxiv.org/abs/2206.07682)

### Flash Attention

[paper](https://arxiv.org/abs/2205.14135)

### Grouped-query attention

[paper](https://arxiv.org/abs/2305.13245)

### ALiBi position encoding

[paper](https://arxiv.org/abs/2108.12409)

### DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale

[paper](https://arxiv.org/abs/2207.00032)

### Claude 1

- Initial release focusing on helpfulness and harmlessness

### FLAN (Fine-tuned LAnguage Net) (Google)

[paper](https://arxiv.org/abs/2109.01652)

- Instruction tuning across multiple tasks
- Improved zero-shot performance

### Red Teaming Language Models with Language Models

[paper](https://arxiv.org/abs/2202.03286)

### HELM (Holistic Evaluation of Language Models)

[paper](https://arxiv.org/abs/2211.09110)

Comprehensive benchmark suite for LLMs
Standardized evaluation metrics

### DALL-E 2 (OpenAI)

- Released in April 2022
- Significant improvement over original DALL-E
- Demonstrated remarkably detailed text-to-image generation
- Maintained controlled access with gradual rollout

### Stable Diffusion (Stability AI)

- Released in August 2022 as "a deep learning, text-to-image model" that became "the premier product of Stability AI"
- Open-source alternative to DALL-E 2
- Democratized access to high-quality image generation
- Trained on LAION-5B dataset

### GPTQ

[paper](https://arxiv.org/abs/2210.17323)

### Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models

[paper](https://arxiv.org/abs/2206.04615)

## 2023: Multi-Modal and Reasoning

### Efficient Memory Management for Large Language Model Serving with PagedAttention

[paper](https://arxiv.org/abs/2309.06180)

### QLoRA: Efficient Finetuning of Quantized LLMs

[paper](https://arxiv.org/abs/2305.14314)

### Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment

[paper](https://arxiv.org/abs/2312.12148)

### FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

[paper](https://arxiv.org/abs/2307.08691)

### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

[paper](https://arxiv.org/abs/2306.00978)

### Generative Agents: Interactive Simulacra of Human Behavior

[paper](https://arxiv.org/abs/2304.03442)

### Voyager: An Open-Ended Embodied Agent with Large Language Models

[paper](https://arxiv.org/abs/2305.16291)

### Universal and Transferable Adversarial Attacks on Aligned Language Models

[paper](https://arxiv.org/abs/2307.15043)

### Towards Monosemanticity: Decomposing Language Models With Dictionary Learning

[paper](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)

### Tree of Thoughts: Deliberate Problem Solving with Large Language Models

[paper](https://arxiv.org/abs/2305.10601)

### Mpt

[blog](https://www.databricks.com/blog/mpt-7b)

### WizardLM: Empowering Large Language Models to Follow Complex Instructions

[paper](https://arxiv.org/abs/2304.12244)

### DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales

[paper](https://arxiv.org/abs/2308.01320)

### GPT-4

[paper](https://arxiv.org/abs/2303.08774)

- Multi-modal encoders
- System prompting
- Advanced reasoning capabilities
- Tool use

### Mistral 7b

[paper](https://arxiv.org/abs/2310.06825)

### LLaMA

[paper](https://arxiv.org/abs/2302.13971)

- Efficient scaling
- Flash Attention-2
- Chat templates
- RLHF improvements

"""
Link: https://huggingface.co/docs/transformers/main/model_doc/llama
Family: Transformer
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: LLaMA uses a Transformer architecture, and with extensions: Pre-normalization, SwiGLU activations, RoPE embeddings, reduced memory usage and runtime through efficient implementation of the causal multi-head attention, checkpointing to reduce the amount of activations that are recomputed during the backward pass, model and sequence parallelism to reduce memory usage of the model, and uses 1.4T BPE tokens after tokenization.
Application: Zero and few shot Commonsense reasoning, Question answering, Code generation and Reading comprehension.
Date (of first known publication): 02/2023
Num. Params: 7B, 13B, 33B and 65B
Corpus: English CommonCrawl + C4 + Github + Wikipedia + Gutenberg and Books3 + ArXiv + Stack Exchange
License: Limited, Non-commercial bespoke license
Lab: Meta
"""

### Mixtral 8x7B

### LLaMA 2

### MamBa

### Alpaca

    """
    Link: https://github.com/tatsu-lab/stanford_alpaca
    Family: LLaMA
    Pretraining Architecture: Decoder
    Fine-tuning Task: human instructions
    Extension: Alpaca is fine-tuned from a 7B LLaMA model.
    Application: Evaluated on a variety of text generation and classification tasks.
    Date (of first known publication): 03/2023
    Num. Params: 7B
    Corpus: 52K instruction-following data generated using self-instruct mechanism, from 175 human-written instruction-output pairs.
    License: Limited, Non-commercial bespoke license
    Lab: Stanford
    """

### Direct Preference Optimization (DPO)

[paper](https://arxiv.org/abs/2305.18290)

### Constitutional AI

[blog](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)

### PaLM 2

[paper](https://arxiv.org/abs/2305.10403)

- Improved multilingual capabilities
- Enhanced reasoning

### LAION-5B (LAION)

[paper](https://arxiv.org/abs/2210.08402)

- Large-scale image-text dataset
- Enabled better multimodal training

### Vicuna (LMSYS)

- Fine-tuned LLaMA
- Open-source conversational agent

### Alpaca (Stanford)

- Instruction-tuned LLaMA
- Efficient fine-tuning approach

### LIMA

[paper](https://arxiv.org/abs/2305.11206)

Demonstrated efficiency of small high-quality datasets
1,000 examples for alignment

### Mamba

[paper](https://arxiv.org/abs/2312.00752)

- State space model for sequence modeling
- Linear scaling with sequence length

### LLaVA (Visual Instruction Tuning)

[paper](https://arxiv.org/abs/2304.08485)

- Released in April 2023 LLaVA was among the first vision-language models created using visual instruction tuning
- Combined vision encoders with language models
- Pioneered efficient visual instruction tuning
- Set foundation for open-source multimodal models

### Claude 1/Claude 2

- Released in March 2023 (Claude 1) and July 2023 (Claude 2)
- Focused on constitutional AI approach
- Enhanced safety and alignment
- Specialized in long-form content generation

### Gemini

- Announced initially in May 2023, fully released in December Described as "a family of multimodal large language models developed by Google DeepMind, and the successor to LaMDA and PaLM 2"
- Designed from the ground up as a multimodal model
- Positioned as Google's answer to GPT-4

### Toy Models of Superposition

[blog](https://transformer-circuits.pub/2022/toy_model/index.html)

### Minerva

[blog](https://research.google/blog/minerva-solving-quantitative-reasoning-problems-with-language-models/)

{IG qwen and deepseek come here}

{Do I include VLMs? Where?}

## 2024: Efficiency and Performance

### Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference

[paper](https://arxiv.org/abs/2403.04132)

### TinyLlama: An Open-Source Small Language Model

[paper](https://arxiv.org/abs/2401.02385)

### MordernBert

### Jamba: A Hybrid Transformer-Mamba Language Model

[paper](https://arxiv.org/abs/2403.19887)

### Gemma

[paper](Gemma: Open Models Based on Gemini Research and Technology)

- Efficient attention mechanisms
- Advanced position embeddings
- Improved tokenization
- Memory efficient training

### Claude 3

[Technical Report](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)

- Multi-modal understanding
- Tool use capabilities
- Advanced reasoning
- Constitutional AI improvements

### LLaMA 3

[paper](https://arxiv.org/abs/2407.21783)

{add quen and deepseek}

### Claude 3

Opus, Sonnet, and Haiku variants
Improved reasoning and multimodal capabilities

### phi-1/phi-2/phi-3

Small but powerful models
High performance with limited training data

### OpenAI o1

First specialized reasoning model
Advanced mathematical problem-solving

### RSO (Reinforced Self-training with Online feedback)

- Self-improvement through AI evaluation
- Reduced human annotation needs

### SPIN (Self-Played Improvement Narration)

- Self-correction capabilities
- Improved factual accuracy

### DBRX

[blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)

### FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

[paper](https://arxiv.org/abs/2407.08608)

### Qwen 2.5 (Alibaba)

- Released in September 2024 as "the latest addition to the Qwen family," which the developers called "the largest opensource release in history"
- Specialized variants for coding and mathematics
- Sizes ranging from 1.5B to 72B parameters
- Strong multilingual capabilities

### DeepSeek 2.5 (DeepSeek)

- Released in September 2024 combining "DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct" as an "upgraded version"
- Competitive code generation capabilities
- Cost-effective alternative to larger models
- 128K token context window

### Claude 3.5 Sonnet (Anthropic)

- Released in October 2024 featuring improved performance "in undergraduate knowledge, graduate-level reasoning, general reasoning, and code generation"
- Advanced reasoning and coding capabilities
- Introduces Artifacts for interactive content creation
- Significant improvements over Claude 3 Opus

### DeepSeek-R1 (DeepSeek)

- Specialized reasoning model released in December 2024
- Focus on mathematical and logical reasoning
- Designed to compete with OpenAI's o1
- Significantly faster inference than o1

### vLLM[], DeepSpeed

## 2025

### Llama 4

### Qwen

### DeepSeek

(there were some amazing developments on tts, video gen, image gen etc but all of those for a different video)

### Grok

- Open-source model
- 314B parameters

### Pixtral

[paper](https://arxiv.org/abs/2410.07073)

- Multimodal capabilities
- 12B parameters

### Qwen2

[paper]()

- Multilingual capabilities
- 72B parameters

### phi

[paper]()

Visual Elements

Add performance charts showing scaling laws
Include architecture diagrams for key innovations
Create a "family tree" showing model lineage

NOTES TO SELF

- Add a note for hardware, not in the scope of this blog but should not be ignored [DONE]
- Quick note about benchmark, Not hear to explain these but these are the major ones that are used mostly.
- Train a hand drawn sketch LORA in flux dev for images
- Add a reference section in the end which redirects to the papers, Like latex reference and stuff.

[blog](https://www.darioamodei.com/essay/machines-of-loving-grace) -->
