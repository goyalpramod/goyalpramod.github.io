<!-- ---
layout: blog
title: "Transformers Laid Out"
date: 2025-01-3 12:00:00 +0530
categories: [personal, technology]
image: /assets/transformers_laid_out/meme.png
---

# Notes on RL 


Now I am aware you must have been looking forward to the second part to my CUDA blog (if you haven't read it yet, check it out!), but hey! a man can have varied interests. And I believe if you are trying to be a great ML engineer, developer or are just EXTREMELY enthusiatic about the space. Reinforcement learning must have tickled your brain as well. 

In my opinion If AI is magic to Computer Science, RL is magic to AI. 

The usual school of thoght while talking about RL (or any other ML topic for the most point) is to first lay out a table of content, show what will be, what are the problems yada yada ya. We are gonna do none of that, we are innovative people and we laugh at the face of the old ways. 

So we will do what innovators do, we will think of the simplest problem we can think of, make a few assumptions, try to solve it and slowly make it complex. 

I invite you to read the following work with an OPEN MIND. So let us begin by first framing a problem. 

## The first problem

Let's say you did some odd job for your neighbour and your naive lil self just got his/her first paycheck!

[INSERT_IMAGE]

Now as you are walking the street, you find this amazing place called a "Casino" and they tell you that you can double your money here. So you walk in...

[INSERT_IMAGE]

oh god what is this ungodly place, you are startled with all the bright lights, money flying around, vomit colored carpet. But you are filled with joy, because you are about to double your money!!!

[INSERT_IMAGE_FROM_WIKI_WITH_COMIC]

As you are walking through this labryinth, you discover a fairly simple looking machinery. Well a bunch of them infact lined up one after the other, they are slot machines! 

[INSERT_IMAGE_OF_SELF_AND_SLOT_MACHINES]

You think, maybe you should try your luck here, as they seem simpler then poker and you just need to put money, and get money. 

You try the first machine (lets assume we give a dollar and if we win we get an unspecified amount of money back, if we lose. THE MACHINE EATS OUR MONEY!!!), after 30 tries you find that instead of having more money, you have lost a significant amount. 

This cannot be right, the hording said that the house never cheats (oh you naive kid, if only the world was as innocent as you are). And that I will in fact get double the money. And on an average, if you play enough times, you will lose some but if the above assumption is true you should win more! 

So you start thinking, and looking around, that's when you observe that the man on machine number 3 seems to be winning quite a fair bit. So you wait for him to leave, and once he does. You go to that machine and try it 30 times, low and behold... you have made more money than you started with! that's when you realise... "THE HOUSE IN FACT CHEATS!" (Who would have guessed right?), now you are an interprid person, who decides to fight back this indignance with math and statistics. 

So you formulate how you can win more money. 

Let us assume we have N tries (The amount of money), and we have k options in front of us (The amount of slot machines). We can assume that each of these k options has an expected return, i.e. a mean around which there is some variance, but if played enough times it will converge to its true value. (Regress to the mean, essentially given enough tries the black box will give its average output. Read more here!)

So let's assume this perfect actual value of a slot machine can be represented as Q(a(subscript k)), but the problem is any time we use it, it does not return the perfect Q(a(subscript k)) (Because if it did, everyone will play all the slot machines once and figure out which gives the highest payout and just use that!), so we can create a running q value that is the sum of expected return. q(a(subscript k))

Which we can write as 

q⇤(a) .= E[Rt | At =a] .

The expected return for an action (here an action is you choosing a particular slot machine)

We can write the q value for any nth try as 

qn = r1 + r2 + r3... rn-1/n-1

This can be simplified as 

qn+1 = 1/n summation R from i = 0 to n
[WRITE THE REST OF THE DERIVATION HERE PLEASE AI]

All in all we can estimate the expected value of any slot machine simply by 

NewEstimate = OldEstimate + step*[reward-OldEstimate]

You try this with the 3 machines, find the one which gave you the highest expected reward, made a huge buck and left for home happy. 
You come the next day, only to realise the casinos caught on to what you were doing. So instead of 3 slot machines, they have now 10!!! machines. 

[INSERT_IMAGE]

Your previous method will not work any more, because you will waste a lot of tries just trying to find the optimal solution. 

So you pull out your trusty notebook and start thinking 

"What if we I assume that every machine gives on average 0 returns, and then I will try a machine, keep that estimate. Now that is my highest returning machine at the moment. So I will keep exploiting that and at random times (Lets say eta times) I will explore and try a new machine, if that gives me greater reward than my current estimate for my current machine. I will stick to that!" 

Wow you mad genuis. You write down your forumla as such (Mad genuises need algorithms to work for some reason)

"
Initialize, for a = 1 to k:
Q(a) 0
N(a) 0
Loop forever:
A
⇢ argmaxa Q(a) with probability 1  " (breaking ties randomly)
a random action with probability "
R bandit(A)
N(A) N(A)+1
Q(A) Q(A)+ 1
N(A)
⇥
R  Q(A)
⇤
"
[WRITE THE CODE PLEASE AI]

You keep doing this for a while, but you are not getting the returns you would like, mostly because you are stuck exploiting only a few machines, while there are many more which could have potentially much higher rewards. 

So you put on your thinking cap again. 


[Story line, Create comics! You go to casino, u start losing money, so you make a plan to maximize your reward. The casino people find out and chase you out, but as you are running away, u get stuck in a maze, now with the casino people after you, you must think of a quick plan to escape!! Thats when you recall that your friend Jack gave you a beautiful bot named maurice who you can program to do anything! You can spawn a new maurice only after 5 min of one maurice dying]

-----



These are my personal notes from MSAI635 (Reinforcement Learning) over here in UMD, and as I go through the book RL by richard and Barto 

I will try to explain each chapter as I understood it as well as do the excercises! 


## CHapter 2 k-armed bandits 

Exercise 2.1 In $\varepsilon$-greedy action selection, for the case of two actions and $\varepsilon = 0.5$, what is
the probability that the greedy action is selected?

Ans -> I believe it will be.

$$P(\text{greedy}) = (1-\varepsilon) \cdot 1 + \varepsilon \cdot \frac{1}{n}$$

where $n$ is the number of actions. With probability $(1-\varepsilon)$ we exploit and pick the greedy action for sure (probability 1). With probability $\varepsilon$ we explore, picking uniformly at random among all $n$ actions (including the greedy one), so the greedy action has a $1/n$ chance there too.

Plugging in $\varepsilon = 0.5$, $n = 2$:

$$P(\text{greedy}) = 0.5 \cdot 1 + 0.5 \cdot \frac{1}{2} = 0.5 + 0.25 = 0.75$$

Exercise 2.2: Bandit example Consider a k-armed bandit problem with k = 4 actions,
denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using
"-greedy action selection, sample-average action-value estimates, and initial estimates
of Q1(a) = 0, for all a. Suppose the initial sequence of actions and rewards is A1 = 1,
R1 = 1, A2 = 2, R2 = 1, A3 = 2, R3 = 2, A4 = 2, R4 = 2, A5 = 3, R5 = 0. On some
of these time steps the " case may have occurred, causing an action to be selected at
random. On which time steps did this definitely occur? On which time steps could this
possibly have occurred? ⇤

Ans -> I believe A2 and A5 are when the random epsilon occured as in these times the greedy action was not taken! 

Exercise 2.3 In the comparison shown in Figure 2.2, which method will perform best in
the long run in terms of cumulative reward and probability of selecting the best action?
How much better will it be? Express your answer quantitatively.

Ans -> $\varepsilon = 0.01$ will perform best in the long run.

Given infinite steps, the sample-average estimates converge to the true action values for every method, so eventually all of them "know" which action is best. What differs afterward is purely how often each method exploits vs. explores. In that steady state:

$$P(\text{optimal action}) = (1-\varepsilon) + \varepsilon \cdot \frac{1}{k}$$

With $k = 10$:

- $\varepsilon = 0.1 \Rightarrow 0.9 + 0.01 = 0.91$ (91%)
- $\varepsilon = 0.01 \Rightarrow 0.99 + 0.001 = 0.991$ (99.1%)

$\varepsilon = 0.01$ wins because once it has learned the best arm, it wastes far fewer pulls exploring randomly, so it exploits the optimal action a much larger fraction of the time.

The same logic applies to reward. The average value of the best of 10 arms (drawn from $\mathcal{N}(0,1)$) is about $1.54$, while a random arm averages $0$. So asymptotic average reward:

- $\varepsilon = 0.1 \Rightarrow 0.9 \times 1.54 \approx 1.386$
- $\varepsilon = 0.01 \Rightarrow 0.99 \times 1.54 \approx 1.525$

So $\varepsilon = 0.01$ ends up roughly 10% higher in both cumulative reward and probability of picking the optimal action.

Exercise 2.4 If the step-size parameters, ↵n, are not constant, then the estimate Qn is
a weighted average of previously received rewards with a weighting di↵erent from that
given by (2.6). What is the weighting on each prior reward for the general case, analogous
to (2.6), in terms of the sequence of step-size parameters?

Ans -> Starting from the same incremental update as (2.6), but now with a general, possibly
non-constant step size $\alpha_i$ at each step:

$$Q_{n+1} = (1-\alpha_n)Q_n + \alpha_n R_n$$

Unrolling this recursion down to $Q_1$ the same way as in Exercise 2.7 (Step 1), but *without*
collapsing the product of $(1-\alpha_j)$ terms into a single power — since the $\alpha_j$ are not
assumed equal here — gives:

$$Q_{n+1} = \Big[\prod_{i=1}^{n}(1-\alpha_i)\Big]Q_1 + \sum_{i=1}^{n}\alpha_i\Big[\prod_{j=i+1}^{n}(1-\alpha_j)\Big]R_i$$

This is the general analogue of (2.6): the weight on $Q_1$ is the product of $(1-\alpha_i)$ over
*all* steps, and the weight on each reward $R_i$ is $\alpha_i$ times the product of $(1-\alpha_j)$
over every step *after* $i$. Setting every $\alpha_i = \alpha$ (constant) collapses each product into
$(1-\alpha)^{n-i}$ and recovers (2.6) exactly — that simplification is only valid in the constant
step-size case, which is why it can't appear in the general answer here.

Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
diculties that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the q⇤(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean 0
and standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like Figure 2.2
for an action-value method using sample averages, incrementally computed, and another
action-value method using a constant step-size parameter, ↵ =0.1. Use " =0.1 and
longer runs, say of 10,000 steps.


```python
import numpy as np
import random

rng = np.random.default_rng()
epsilon = 0.1

def step(q):
    mean = 0
    std_dev = 0.01
    
    value = rng.normal(loc=mean, scale=std_dev, size=10)
    q += value
    return q

class Agent:
    def __init__(self, k=10):
        self.q = [0 for i in range(k)]
        self.n = [0 for i in range(k)]

    def sample_average(self, action):
        self.n[action] += 1
        q_star = rng.normal(loc=Q[action] , scale=1)
        self.q[action] += (1/(self.n[action]))*(q_star - self.q[action])
        return self.q[action]

    def constant_step_size(self, action, alpha = 0.1):
        self.n[action] += 1
        q_star = rng.normal(loc=Q[action] , scale=1)
        self.q[action] += (alpha)*(q_star - self.q[action])
        return self.q[action]

reward_sum_1 = np.zeros(10000)
reward_sum_2 = np.zeros(10000)
optimal_count_1 = np.zeros(10000)
optimal_count_2 = np.zeros(10000)

for j in range(2000):
    agent_1 = Agent();
    agent_2 = Agent();

    Q = [5.0]*10
    Q = np.asarray(Q)
    
    for i in range(10000):
        Q = step(Q)
        value_1 = random.random();
        value_2 = random.random();

        if(value_1 > epsilon):
            action_1 = agent_1.q.index(max(agent_1.q))
        else:
            action_1 = random.randint(0, 9)
        
        if(value_2 > epsilon):
            action_2 = agent_2.q.index(max(agent_2.q))
        else:
            action_2 = random.randint(0, 9)

        reward_1 = agent_1.sample_average(action_1)
        reward_2 = agent_2.constant_step_size(action_2)

        reward_sum_1[i] += reward_1
        reward_sum_2[i] += reward_2

        if (action_1 == Q.argmax()):
            optimal_count_1[i] += 1
        else:
            continue
            
        if (action_2 == Q.argmax()):
            optimal_count_2[i] += 1
        else:
            continue

avg_reward_1 = reward_sum_1 / 2000
pct_optimal_1 = optimal_count_1 / 2000 * 100

avg_reward_2 = reward_sum_2 / 2000
pct_optimal_2 = optimal_count_2 / 2000 * 100
```

```python
import matplotlib.pyplot as plt

plt.plot(avg_reward_1, label="sample average")
plt.plot(avg_reward_2, label="constant step-size")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.legend()
plt.show()
```


Exercise 2.6: Mysterious Spikes The results shown in Figure 2.3 should be quite reliable
because they are averages over 2000 individual, randomly chosen 10-armed bandit tasks.
Why, then, are there oscillations and spikes in the early part of the curve for the optimistic
method? In other words, what might make this method perform particularly better or
worse, on average, on particular early steps?

Ans -> As the number of bandit is  restricted to 10, the optimistic one is going to try all 10 of them, and one of them is likely to be the most optimal action, and equally one with the least optimal action. That is why we see oscilations early on. 


Exercise 2.7: Unbiased Constant-Step-Size Trick In most of this chapter we have used
sample averages to estimate action values because sample averages do not produce the
initial bias that constant step sizes do (see the analysis leading to (2.6)). However, sample
averages are not a completely satisfactory solution because they may perform poorly
on nonstationary problems. Is it possible to avoid the bias of constant step sizes while
retaining their advantages on nonstationary problems? One way is to use a step size of

$$\beta_n \doteq \alpha / \bar{o}_n \tag{2.8}$$

to process the $n$th reward for a particular action, where $\alpha > 0$ is a conventional constant
step size, and $\bar{o}_n$ is a trace of one that starts at 0:

$$\bar{o}_n \doteq \bar{o}_{n-1} + \alpha(1 - \bar{o}_{n-1}), \quad \text{for } n > 0, \text{ with } \bar{o}_0 \doteq 0 \tag{2.9}$$

Carry out an analysis like that in (2.6) to show that $Q_n$ is an exponential recency-weighted
average *without initial bias*.

Ans -> Start from the incremental update with step size $\beta_n$:

$$Q_{n+1} = Q_n + \beta_n\left[R_n - Q_n\right] = (1-\beta_n)Q_n + \beta_n R_n$$

Unrolling this recursion one level at a time:

$$Q_{n+1} = (1-\beta_n)\left[(1-\beta_{n-1})Q_{n-1} + \beta_{n-1}R_{n-1}\right] + \beta_n R_n$$

$$= (1-\beta_n)(1-\beta_{n-1})(1-\beta_{n-2})Q_{n-2} + (1-\beta_n)(1-\beta_{n-1})\beta_{n-2}R_{n-2} + (1-\beta_n)\beta_{n-1}R_{n-1} + \beta_n R_n$$

Continuing all the way down to $Q_1$, the pattern is:

$$Q_{n+1} = \Big[\prod_{j=1}^{n}(1-\beta_j)\Big]Q_1 + \sum_{i=1}^{n}\beta_i\Big[\prod_{j=i+1}^{n}(1-\beta_j)\Big]R_i$$

Now express $1-\beta_j$ in terms of $\bar{o}$. Since $\beta_j = \alpha/\bar{o}_j$:

$$1-\beta_j = \frac{\bar{o}_j - \alpha}{\bar{o}_j}$$

From (2.9), $\bar{o}_j = \bar{o}_{j-1} + \alpha(1-\bar{o}_{j-1})$, so $\bar{o}_j - \alpha = \bar{o}_{j-1} - \alpha\bar{o}_{j-1} = (1-\alpha)\bar{o}_{j-1}$. Therefore:

$$1-\beta_j = \frac{(1-\alpha)\,\bar{o}_{j-1}}{\bar{o}_j}$$

**The coefficient on $Q_1$.** The product telescopes, because each $\bar{o}_j$ in a denominator cancels against the numerator of the next factor:

$$\prod_{j=1}^{n}\frac{(1-\alpha)\,\bar{o}_{j-1}}{\bar{o}_j} = (1-\alpha)^n\cdot\frac{\bar{o}_0}{\bar{o}_1}\cdot\frac{\bar{o}_1}{\bar{o}_2}\cdots\frac{\bar{o}_{n-1}}{\bar{o}_n} = (1-\alpha)^n\,\frac{\bar{o}_0}{\bar{o}_n} = 0$$

since $\bar{o}_0 = 0$. The initial estimate $Q_1$ vanishes entirely, which means there is no initial bias.

**The coefficient on $R_i$.** The same telescoping applies over $j = i+1, \dots, n$:

$$\beta_i\prod_{j=i+1}^{n}(1-\beta_j) = \frac{\alpha}{\bar{o}_i}\cdot(1-\alpha)^{n-i}\cdot\frac{\bar{o}_i}{\bar{o}_n} = \frac{\alpha(1-\alpha)^{n-i}}{\bar{o}_n}$$

**Result:**

$$Q_{n+1} = \frac{1}{\bar{o}_n}\sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i$$

The weight on $R_i$ is proportional to $(1-\alpha)^{n-i}$, so it decays exponentially as the reward gets older, exactly like (2.6), but there is no $(1-\alpha)^n Q_1$ term.

The weights also sum to 1. From (2.9), $\bar{o}_n - 1 = (1-\alpha)(\bar{o}_{n-1} - 1)$ with $\bar{o}_0 - 1 = -1$, which gives $\bar{o}_n = 1-(1-\alpha)^n$. Then:

$$\sum_{i=1}^{n}\alpha(1-\alpha)^{n-i} = \alpha\cdot\frac{1-(1-\alpha)^n}{\alpha} = 1-(1-\alpha)^n = \bar{o}_n$$

so dividing by $\bar{o}_n$ normalizes the weights to exactly 1. Sanity check for $n=1$: $\bar{o}_1 = \alpha$, so $\beta_1 = 1$ and $Q_2 = R_1$, which matches the formula.

Hence $Q_n$ is an exponential recency-weighted average without initial bias.



Exercise 2.8: UCB Spikes In Figure 2.4 the UCB algorithm shows a distinct spike
in performance on the 11th step. Why is this? Note that for your answer to be fully
satisfactory it must explain both why the reward increases on the 11th step and why it
decreases on the subsequent steps. Hint: If c = 1, then the spike is less prominent. 

Ans -> The UCB action-selection rule is:

$$A_t \doteq \arg\max_a \left[Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}\right]$$

where any action with $N_t(a) = 0$ is treated as maximizing, forcing it to be selected. So over
the first 10 steps, UCB is forced to try each of the 10 arms exactly once (a round-robin), since
an untried arm always wins the argmax regardless of its $Q$ estimate. Going into step 11,
$N_{11}(a) = 1$ for every arm.

**Why the spike at step 11.** Since $N_{11}(a) = 1$ for all $a$, the exploration bonus
$c\sqrt{\ln(11)/N_{11}(a)} = c\sqrt{\ln 11}$ is identical for every arm, so it contributes nothing to
breaking the tie. Selection at step 11 therefore collapses to $\arg\max_a Q_{11}(a)$ — purely
greedy, based on the single (possibly noisy) reward each arm produced during its one earlier pull.
An arm's one-shot sample is more likely to be large if its true mean is large, so this greedy pick
has a decent chance of landing on the actually-best arm — much better odds than the forced
round-robin of steps 1-10, which guaranteed several genuinely suboptimal arms got pulled. That
is what produces the reward spike right at step 11.

**Why it drops again afterward.** Once step 11 happens, the chosen arm's count increases to
$N(a) = 2$, while the other 9 arms remain at $N(a) = 1$ — the tie is broken. The just-picked arm's
exploration bonus shrinks, while the other 9 arms still carry the larger bonus that comes from
having only 1 sample. UCB now favors re-exploring those 9 under-sampled arms over continuing to
exploit the good pick from step 11. Since each of those arms' $Q$-estimates is also based on just
a single noisy sample, several of them are likely to look worse than they truly are (an unlucky
first draw). Re-selecting them pulls the average reward back down right after the peak, producing
the spike-then-dip shape.

**Why $c=1$ makes the spike less prominent.** The round-robin phase (steps 1-10) happens
regardless of $c$, since untried arms are always treated as maximizing no matter how small $c$ is.
At step 11 itself, the tie-break is purely on $Q$, since the (equal) bonus term cancels out of the
comparison — so $c$ doesn't affect the quality of the step-11 pick either. What $c$ *does* control
is how strongly the algorithm swings back toward the 9 once-tried arms afterward: a larger $c$
(e.g. the book's default $c=2$) makes that pull-back strong, producing a sharp dip right after the
peak, hence a pronounced spike shape. With $c=1$, the exploration-bonus differences are smaller,
so the post-peak swing back to exploration is gentler, smoothing out what would otherwise look
like a sharp spike.


Exercise 2.9 Show that in the case of two actions, the soft-max distribution is the same
as that given by the logistic, or sigmoid, function often used in statistics and artificial
neural networks.

Ans -> The soft-max (Gibbs/Boltzmann) distribution over $k$ actions, using preferences $H_t(a)$, is:

$$\pi_t(a) = \Pr\{A_t = a\} = \frac{e^{H_t(a)}}{\sum_{c=1}^{k} e^{H_t(c)}}$$

For two actions $a$ and $b$, this becomes:

$$\pi_t(a) = \frac{e^{H_t(a)}}{e^{H_t(a)} + e^{H_t(b)}}$$

Dividing numerator and denominator by $e^{H_t(a)}$ collapses the two exponential terms into one:

$$\pi_t(a) = \frac{1}{1 + e^{H_t(b) - H_t(a)}}$$

Comparing this to the logistic/sigmoid function $\sigma(x) = \dfrac{1}{1+e^{-x}}$, this is exactly
$\sigma(x)$ with $x = H_t(a) - H_t(b)$:

$$\pi_t(a) = \sigma\big(H_t(a) - H_t(b)\big)$$

So in the two-action case, soft-max action selection reduces exactly to the logistic sigmoid of
the *difference* between the two preferences.

This has a nice consequence: $x$ depends only on the **difference** between $H_t(a)$ and
$H_t(b)$, never on their absolute magnitudes. If a constant $c$ were added to both preferences
($H_t(a)+c$ and $H_t(b)+c$), it would cancel out of $x = H_t(a)-H_t(b)$ entirely, leaving
$\pi_t(a)$ unchanged. Soft-max/Gibbs action selection is therefore shift-invariant — only the
relative gap between preferences drives behavior, not their absolute scale. This is also why the
book can safely initialize every $H_1(a) = 0$ without biasing action selection: all actions start
at the same absolute preference, and only the differences that develop over time end up mattering.

Exercise 2.10 Suppose you face a 2-armed bandit task whose true action values change
randomly from time step to time step. Specifically, suppose that, for any time step,
the true values of actions 1 and 2 are respectively 10 and 20 with probability 0.5 (case
A), and 90 and 80 with probability 0.5 (case B). If you are not able to tell which case
you face at any step, what is the best expected reward you can achieve and how should
you behave to achieve it? Now suppose that on each step you are told whether you are
facing case A or case B (although you still don’t know the true action values). This is an
associative search task. What is the best expected reward you can achieve in this task,
and how should you behave to achieve it?


Ans -> **Part 1: no information about which case is active.**

If you always pick action 1 regardless of case, the expected reward is:

$$0.5 \times 10 + 0.5 \times 90 = 50$$

If you always pick action 2 regardless of case, the expected reward is:

$$0.5 \times 20 + 0.5 \times 80 = 50$$

Both fixed actions give exactly $50$ in expectation. Since the case flips independently and
randomly on every single step with no way to observe or predict which one is active before
acting, there is no information a learning algorithm — sample-average, constant step-size, or
anything else — could extract that would let it do better on any particular step than picking
blindly. Tracking $Q(1)$ and $Q(2)$ over many steps would just converge both estimates toward
$50$, and knowing that doesn't help you choose better on the next, individual step, since case A
and case B are equally likely regardless of history. So the best achievable expected reward here
is $\mathbf{50}$, and *any* fixed choice (or even choosing randomly) achieves it — there's no
advantage to switching, hedging, or using an adaptive algorithm, because there's nothing
exploitable in the environment to adapt to.

**Part 2: told which case you're facing (associative search).**

Now the case label is given before you act, even though the underlying values (10/20 in A,
90/80 in B) still have to be behaved toward correctly. The best strategy is to condition the
action on the case:

- In case A, action 2 is better ($20 > 10$) — pick action 2.
- In case B, action 1 is better ($90 > 80$) — pick action 1.

Best expected reward:

$$0.5 \times 20 + 0.5 \times 90 = 10 + 45 = \mathbf{55}$$

This is strictly better than the no-information case, because now the case label lets you
condition your policy on context — learning (or simply knowing) the best action *per case*
rather than being forced into a single unconditional choice. This is the essence of an
associative search / contextual bandit task: the extra information turns "one best action
overall" into "one best action per context," which is a strictly easier and more rewarding
problem to solve.

Exercise 2.11 (programming) Make a figure analogous to Figure 2.6 for the nonstationary
case outlined in Exercise 2.5. Include the constant-step-size $\varepsilon$-greedy algorithm with
$\alpha=0.1$. Use runs of 200,000 steps and, as a performance measure for each algorithm and
parameter setting, use the average reward over the last 100,000 steps.

Ans -> This reuses the nonstationary environment from Exercise 2.5 (all $q_*(a)$ start equal and
independently random-walk with std $0.01$ per step), but now instead of a single run per method,
it's a full parameter study like Figure 2.6: four algorithm families, each swept over a range of
its own parameter, each point evaluated as the average reward over the *last* 100,000 of a
200,000-step run (not the first 1,000 like the original Figure 2.6 — since values keep drifting
forever here, steady-state performance under drift is what matters, not early transient behavior).

The key change from the stationary Figure 2.6: every value-based method (ε-greedy, UCB,
optimistic greedy) uses the **constant step-size** update ($\alpha=0.1$) instead of sample
averages, since Exercise 2.5 already showed sample averages can't track drifting values. The
gradient bandit method is unaffected by this — it never maintains $Q$ estimates at all, only
preferences $H$, so its own step-size parameter is what gets swept instead.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

k = 10
n_steps = 200_000
n_runs = 50          # book-quality smoothness needs ~2000 runs; this is expensive at
                      # 200,000 steps each, so start small and scale up if you have the compute
last_n = 100_000

def walk(q, std_dev=0.01):
    return q + rng.normal(0, std_dev, size=k)

def run_epsilon_greedy(epsilon, alpha=0.1):
    total = 0.0
    for _ in range(n_runs):
        q_true = np.zeros(k)
        Q = np.zeros(k)
        for t in range(n_steps):
            q_true = walk(q_true)
            a = rng.integers(k) if rng.random() < epsilon else np.argmax(Q)
            r = rng.normal(q_true[a], 1)
            Q[a] += alpha * (r - Q[a])
            if t >= n_steps - last_n:
                total += r
    return total / (n_runs * last_n)

def run_ucb(c, alpha=0.1):
    total = 0.0
    for _ in range(n_runs):
        q_true = np.zeros(k)
        Q = np.zeros(k)
        N = np.zeros(k)
        for t in range(n_steps):
            q_true = walk(q_true)
            a = np.argmin(N) if np.any(N == 0) else np.argmax(Q + c * np.sqrt(np.log(t + 1) / N))
            N[a] += 1
            r = rng.normal(q_true[a], 1)
            Q[a] += alpha * (r - Q[a])
            if t >= n_steps - last_n:
                total += r
    return total / (n_runs * last_n)

def run_optimistic_greedy(Q0, alpha=0.1):
    total = 0.0
    for _ in range(n_runs):
        q_true = np.zeros(k)
        Q = np.full(k, float(Q0))
        for t in range(n_steps):
            q_true = walk(q_true)
            a = np.argmax(Q)
            r = rng.normal(q_true[a], 1)
            Q[a] += alpha * (r - Q[a])
            if t >= n_steps - last_n:
                total += r
    return total / (n_runs * last_n)

def run_gradient_bandit(alpha):
    total = 0.0
    for _ in range(n_runs):
        q_true = np.zeros(k)
        H = np.zeros(k)
        avg_reward = 0.0
        for t in range(n_steps):
            q_true = walk(q_true)
            exp_H = np.exp(H - H.max())
            pi = exp_H / exp_H.sum()
            a = rng.choice(k, p=pi)
            r = rng.normal(q_true[a], 1)

            one_hot = np.zeros(k)
            one_hot[a] = 1
            H += alpha * (r - avg_reward) * (one_hot - pi)   # baseline uses reward *before* this step
            avg_reward += (r - avg_reward) / (t + 1)

            if t >= n_steps - last_n:
                total += r
    return total / (n_runs * last_n)

epsilons = [2.0**e for e in range(-7, -1)]      # 1/128 ... 1/4
alphas_grad = [2.0**e for e in range(-5, 3)]    # 1/32 ... 4
cs = [2.0**e for e in range(-4, 3)]             # 1/16 ... 4
Q0s = [2.0**e for e in range(-2, 3)]            # 1/4 ... 4

eps_perf = [run_epsilon_greedy(e) for e in epsilons]
grad_perf = [run_gradient_bandit(a) for a in alphas_grad]
ucb_perf = [run_ucb(c) for c in cs]
opt_perf = [run_optimistic_greedy(q0) for q0 in Q0s]

plt.figure(figsize=(10, 6))
plt.plot(np.log2(epsilons), eps_perf, marker='o', label=r'$\varepsilon$-greedy, constant $\alpha=0.1$')
plt.plot(np.log2(alphas_grad), grad_perf, marker='o', label='gradient bandit')
plt.plot(np.log2(cs), ucb_perf, marker='o', label=r'UCB, constant $\alpha=0.1$')
plt.plot(np.log2(Q0s), opt_perf, marker='o', label=r'optimistic greedy, constant $\alpha=0.1$')
plt.xlabel(r'Parameter ($2^x$, log scale)')
plt.ylabel('Average reward over last 100,000 steps')
plt.title('Figure 2.6 analogue — nonstationary 10-armed testbed')
plt.legend()
plt.show()
```

**Why this design:**
- **`walk`** is the same random-walk step from Exercise 2.5, called once per time step regardless of which agent is running, so every method faces an identically-shaped drifting environment.
- **UCB's tie-break for untried arms** (`np.argmin(N)` when any $N(a)=0$) mirrors the forced round-robin from Exercise 2.8 — every arm has to be tried once before the UCB bonus formula is even well-defined ($N(a)=0$ would divide by zero otherwise).
- **The gradient bandit's baseline** must use the running average *before* it's updated with the current reward — updating the baseline first and then using it would let the current reward leak into its own baseline, biasing the preference update.
- **Runtime warning:** this is a genuinely expensive simulation — `n_runs=50` here already means $50 \times 200{,}000 \approx 10^7$ steps *per parameter value*, times roughly 30 parameter values across the four families. Expect this to take a while in pure Python; drop `n_runs` further for a quick sanity check, or vectorize the inner loop across runs with NumPy arrays instead of a Python `for` loop over `n_runs` if you want it to run faster at higher fidelity.

## Chapter 3 Finite Markov Decision Processes

**Note to self — conditional expectation vs. conditional probability:** these get confused easily since both use "$\mid$", but they're different objects.

- Conditional *probability*, $P(X=x\mid Y=y)$, answers "given $Y=y$, what's the probability $X=x$?" — a number between 0 and 1.
- Conditional *expectation*, $\mathbb{E}[X\mid Y=y]$, answers "given $Y=y$, what's the average value $X$ takes?" — it's not a probability, it can be any real number (whatever units $X$ is in).

The link: conditional probabilities are the *weights* used to compute a conditional expectation — $\mathbb{E}[X\mid Y=y] = \sum_x x \cdot P(X=x\mid Y=y)$ — but the expectation's output is a value, not a probability.

Example (non-RL): $X$ = height of a random person, $Y$ = their country. $P(X=180\text{cm}\mid Y=\text{Netherlands})$ is a probability (e.g. $0.04$). $\mathbb{E}[X\mid Y=\text{Netherlands}]$ is "average height of Dutch people" (e.g. $183$cm) — a height, not a probability.

So in something like $v_\pi(s) = \mathbb{E}_\pi[q_\pi(s,A_t)\mid S_t=s]$: $q_\pi(s,A_t)$ is already a *value* (an expected return), $A_t$ is the random variable being averaged over, and its conditional distribution $\pi(a\mid s)$ supplies the weights — the output $v_\pi(s)$ is a value, not a probability.

Exercise 3.1 Devise three example tasks of your own that fit into the MDP framework,
identifying for each its states, actions, and rewards. Make the three examples as di↵erent
from each other as possible. The framework is abstract and flexible and can be applied in
many di↵erent ways. Stretch its limits in some way in at least one of your examples. ⇤

Ans. 
1. A pizza making robot, the state can be the current state of the pizza and the actions can be the amount of ingredients to put and the reward will be if the user liked it or not 

2. A Water heater, the action is heating up the copper wire, state is the current temperature of the water, reward is how close is it to the expected water temperature 

3. Teaching a one legged robot to walk, the motors can be the action, the current position and if it is upright is state, the distance traveled is the reward.


Exercise 3.2 Is the MDP framework adequate to usefully represent all goal-directed
learning tasks? Can you think of any clear exceptions? ⇤

Ans. No it is not, as it expects that the current state depends exclusively on the previous state and disregards all the history prior to that. This can fail in tasks where all the states are important. An example can be equity bot which sells equity, the worth of an equity cannot be directly valued by what it was in the previous state, we also have to look at when it was bought and for how much, depending on a prior step!


Exercise 3.3 Consider the problem of driving. You could define the actions in terms of
the accelerator, steering wheel, and brake, that is, where your body meets the machine.
Or you could define them farther out—say, where the rubber meets the road, considering
your actions to be tire torques. Or you could define them farther in—say, where your
brain meets your body, the actions being muscle twitches to control your limbs. Or you
could go to a really high level and say that your actions are your choices of where to drive.
What is the right level, the right place to draw the line between agent and environment?
On what basis is one location of the line to be preferred over another? Is there any
fundamental reason for preferring one location over another, or is it a free choice? 

Ans. Let us look at first what does not work. 

Rubber meets the road does not work obviously as we have no control over that and we can not have well defined actions over it! 

Brain meets body, well it does provide actions it cannot be constratined. It has far too many variables. 

Where to drive is not the right level as well, because the destination is constant the path to reach it are multiple. We have to define a level in which we have control, defined variables, and something that we can optimize. 

After the process of elimination the only reasonable answer left is actions in terms of brakes, acceleration...


Exercise 3.4 Give a table analogous to that in Example 3.3, but for $p(s', r \mid s, a)$. It
should have columns for $s$, $a$, $s'$, $r$, and $p(s', r \mid s, a)$, and a row for every 4-tuple for which
$p(s', r \mid s, a) > 0$.

Ans -> Example 3.3 is the recycling robot: states $\{high, low\}$, actions $search$ and $wait$
(available in both states), and $recharge$ (available only in $low$), with $\alpha$ the probability
of staying at $high$ after searching from $high$, $\beta$ the probability of staying at $low$
after searching from $low$, and expected rewards $r_{search}$, $r_{wait}$, and $-3$ (for running
out of charge and needing rescue). Its table gives $p(s'\mid s,a)$ and $r(s,a,s')$ separately;
here, since the reward is deterministic given each transition, $p(r\mid s',s,a)=1$ for that one
$r$ and $0$ otherwise, so $p(s',r\mid s,a)$ just equals $p(s'\mid s,a)$ dropped into the row for
its corresponding deterministic $r$:

| $s$ | $a$ | $s'$ | $r$ | $p(s',r\mid s,a)$ |
|---|---|---|---|---|
| high | search | high | $r_{search}$ | $\alpha$ |
| high | search | low | $r_{search}$ | $1-\alpha$ |
| low | search | high | $-3$ | $1-\beta$ |
| low | search | low | $r_{search}$ | $\beta$ |
| high | wait | high | $r_{wait}$ | $1$ |
| low | wait | low | $r_{wait}$ | $1$ |
| low | recharge | high | $0$ | $1$ |

Every other $(s,a,s',r)$ combination not listed here has $p(s',r\mid s,a)=0$ — e.g. $(low,
recharge, low, \cdot)$, since $recharge$ always succeeds and moves the robot to $high$ with
certainty, so $p(low \mid low, recharge) = 0$.

Exercise 3.5 The equations in Section 3.1 are for the continuing case and need to be
modified (very slightly) to apply to episodic tasks. Show that you know the modifications
needed by giving the modified version of (3.3).

Ans. The asymmetry is between the two occurrences of "state" in the equation: $s'$ is
something being transitioned *into*, so it's fine for it to land on the terminal state —
an episode ending is a perfectly normal transition outcome. But $s$ is something being
acted *from*, via $\mathcal{A}(s)$, and "the set of actions available at the terminal
state" doesn't mean anything, since the agent never acts once the episode has ended.
So only the $s'$ side needs to widen to a set that includes the terminal state, call it
$\mathcal{S}^+ = \mathcal{S} \cup \{\text{terminal}\}$, while $s$ stays restricted to
$\mathcal{S}$ (the nonterminal states):

$$\sum_{s' \in \mathcal{S}^+} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) = 1, \quad \text{for all } s \in \mathcal{S}, a \in \mathcal{A}(s)$$

Exercise 3.6 Suppose you treated pole-balancing as an episodic task but also used
discounting, with all rewards zero except for -1 upon failure. What then would the
return be at each time? How does this return di↵er from that in the discounted, continuing
formulation of this task? ⇤

[Ans.] Since every reward is $0$ except $-1$ at the moment of failure, in $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ almost every term vanishes — only the term landing exactly on a failure time survives.

**Episodic case:** there is one failure, at time $T$. The surviving term is where $t+k+1=T$, i.e. $k = T-t-1$, so

$$G_t = -\gamma^{\,T-t-1}$$

a single, exact term determined entirely by how many steps remain until that one termination.

**Continuing case:** the task never stops, so failure (and reset) keeps happening, at times $T_1 < T_2 < T_3 < \dots$ after $t$. Now a nonzero term shows up at *every* one of those failure times, not just the first, so the sum has one term per future failure:

$$G_t = -\sum_{i=1}^{\infty} \gamma^{\,T_i - t - 1}$$

**Difference:** the episodic return is a single discounted term for the one upcoming termination. The continuing return is an infinite sum of such terms, one for every failure the pole will ever have in the future — it only converges to a finite number because $\gamma < 1$ shrinks the contribution of failures far in the future, not because the return is "close to zero."

Exercise 3.7 Imagine that you are designing a robot to run a maze. You decide to give it a
reward of +1 for escaping from the maze and a reward of zero at all other times. The task
seems to break down naturally into episodes—the successive runs through the maze—so
you decide to treat it as an episodic task, where the goal is to maximize expected total
reward (3.7). After running the learning agent for a while, you find that it is showing
no improvement in escaping from the maze. What is going wrong? Have you e↵ectively
communicated to the agent what you want it to achieve? ⇤

[Ans.] No, because the agent has no metric of knowing if it has improved over time. Nor any other incentive to do so.

Exercise 3.8 Suppose  =0.5 and the following sequence of rewards is received R1 = -1,
R2 = 2, R3 = 6, R4 = 3, and R5 = 2, with T = 5. What are G0, G1, ..., G5? Hint:
Work backwards. ⇤

[Ans.] Working backwards with $G_t = R_{t+1} + \gamma G_{t+1}$:

- $G_5 = 0$
- $G_4 = R_5 + \gamma G_5 = 2 + 0.5(0) = 2$
- $G_3 = R_4 + \gamma G_4 = 3 + 0.5(2) = 4$
- $G_2 = R_3 + \gamma G_3 = 6 + 0.5(4) = 8$
- $G_1 = R_2 + \gamma G_2 = 2 + 0.5(8) = 6$
- $G_0 = R_1 + \gamma G_1 = -1 + 0.5(6) = 2$


Exercise 3.9 Suppose  =0.9 and the reward sequence is R1 = 2 followed by an infinite
sequence of 7s. What are G1 and G0? ⇤

[Ans.] $G_1 = R_2 + \gamma R_3 + \gamma^2 R_4 + \dots = 7(1+\gamma+\gamma^2+\dots) = \dfrac{7}{1-\gamma} = \dfrac{7}{0.1} = 70$

$G_0 = R_1 + \gamma G_1 = 2 + 0.9(70) = 2 + 63 = 65$


Exercise 3.10 Prove the second equality in (3.10).
Ans -> Sum of GP

**[IMPORTANT]** Exercise 3.11 If the current state is St, and actions are selected according to a stochastic
policy ⇡, then what is the expectation of Rt+1 in terms of ⇡ and the four-argument
function p (3.2)?

[Ans.] For a fixed action $a$, the expected reward averages $r$ over the joint distribution of $(s',r)$ given by the four-argument $p$:

$$\mathbb{E}[R_{t+1}\mid S_t=s, A_t=a] = \sum_{s'}\sum_{r} r \cdot p(s',r\mid s,a)$$

Since the action itself is random, drawn from $\pi(\cdot\mid s)$, average that over $a$ too:

$$\mathbb{E}[R_{t+1}\mid S_t=s] = \sum_{a} \pi(a\mid s) \sum_{s'}\sum_{r} r \cdot p(s',r\mid s,a)$$


Exercise 3.12 Give an equation for $v_\pi$ in terms of $q_\pi$ and $\pi$.

[Ans.] Start from the definitions (3.12) and (3.13):

$$v_\pi(s) := \mathbb{E}_\pi[G_t \mid S_t=s], \qquad q_\pi(s,a) := \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$$

$v_\pi(s)$ is an expectation over $G_t$ that isn't yet conditioned on which action gets taken — but $A_t$ is itself random, distributed as $\pi(\cdot\mid s)$. Condition on $A_t$ and apply the law of total expectation (i.e. average the *conditional* expectation over the distribution of the thing being conditioned on):

$$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s] = \sum_{a} \Pr(A_t=a \mid S_t=s) \cdot \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$$

$\Pr(A_t=a\mid S_t=s)$ is exactly $\pi(a\mid s)$ by definition of the policy, and the remaining conditional expectation is exactly $q_\pi(s,a)$ by definition. Substituting both in:

$$v_\pi(s) = \sum_{a} \pi(a\mid s)\, q_\pi(s,a)$$

This is the same law-of-total-expectation step used in Exercise 3.11, just with the full return $G_t$ in place of the single reward $R_{t+1}$.

**[IMPORTANT]** Exercise 3.13 Give an equation for $q_\pi$ in terms of $v_\pi$ and the four-argument $p$.

[Ans.] Start from the definition and the recursive relationship between returns (Exercise 3.10):

$$q_\pi(s,a) := \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a] = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid s,a]$$

By linearity of expectation, split this into two terms:

$$q_\pi(s,a) = \mathbb{E}_\pi[R_{t+1}\mid s,a] + \gamma\,\mathbb{E}_\pi[G_{t+1}\mid s,a]$$

**First term.** With $s,a$ fixed, the only remaining randomness in $R_{t+1}$ is which $(s',r)$ outcome occurs, distributed exactly by the four-argument $p$:

$$\mathbb{E}_\pi[R_{t+1}\mid s,a] = \sum_{s'}\sum_r r \cdot p(s',r\mid s,a)$$

**Second term.** $G_{t+1}$ depends on rewards further in the future than $p(s',r\mid s,a)$ describes, so this takes two ideas:

*Markov property:* if $S_{t+1}=s'$ were known, the future is fully determined (probabilistically) by $s'$ alone — knowing $s,a$ in addition gives no extra information, so they can be dropped from the conditioning:

$$\mathbb{E}[G_{t+1}\mid S_{t+1}=s', S_t=s, A_t=a] = \mathbb{E}[G_{t+1}\mid S_{t+1}=s']$$

*Stationarity:* since $\pi$ and the dynamics don't change with the time index, this equals the same quantity at time $t$: $\mathbb{E}[G_{t+1}\mid S_{t+1}=s'] = \mathbb{E}[G_t\mid S_t=s'] = v_\pi(s')$.

But $S_{t+1}$ is random given only $s,a$ — so average $v_\pi(s')$ over its distribution (law of total expectation). That distribution is the *marginal* of the four-argument $p$ over $s'$, obtained by summing out $r$ (same as $P(X=x)=\sum_y P(X=x,Y=y)$ for any joint distribution):

$$\Pr(S_{t+1}=s'\mid s,a) = \sum_r p(s',r\mid s,a)$$

$$\mathbb{E}_\pi[G_{t+1}\mid s,a] = \sum_{s'}\left(\sum_r p(s',r\mid s,a)\right) v_\pi(s')$$

**Combine.** Both terms are sums over the same $(s',r)$ pairs, so merge them into one double sum, factoring out $p(s',r\mid s,a)$:

$$q_\pi(s,a) = \sum_{s'}\sum_r r\cdot p(s',r\mid s,a) + \gamma\sum_{s'}\sum_r p(s',r\mid s,a)\,v_\pi(s')$$

$$q_\pi(s,a) = \sum_{s'}\sum_r p(s',r\mid s,a)\,\big[r + \gamma\, v_\pi(s')\big]$$

Exercise 3.14 The Bellman equation (3.14) must hold for each state for the value function
v⇡ shown in Figure 3.2 (right) of Example 3.5. Show numerically that this equation holds
for the center state, valued at +0.7, with respect to its four neighboring states, valued at
+2.3, +0.4, 0.4, and +0.7. (These numbers are accurate only to one decimal place.) ⇤

[Ans.] Under the equiprobable random policy $\pi(a\mid s)=0.25$ for each of the four actions; each action deterministically moves to one neighboring cell with reward $r=0$; $\gamma=0.9$:

$$v_\pi(s) = \sum_a \pi(a\mid s)\big[r+\gamma v_\pi(s')\big] = 0.25 \times 0.9 \times (2.3+0.4-0.4+0.7)$$

$$= 0.225 \times 3.0 = 0.675 \approx 0.7$$

which matches the stated value of $+0.7$ (to the one-decimal-place accuracy given).

Exercise 3.15 In the gridworld example, rewards are positive for goals, negative for
running into the edge of the world, and zero the rest of the time. Are the signs of these
rewards important, or only the intervals between them? Prove, using (3.8), that adding a
constant c to all the rewards adds a constant, vc, to the values of all states, and thus
does not a↵ect the relative values of any states under any policies. What is vc in terms
of c and ? ⇤

[Ans.] Only the intervals between rewards matter, not their absolute signs — adding a constant $c$ shifts every state's value by the same fixed amount, so the relative ordering of states (and hence which policy is better than which) is unchanged.

Proof, using (3.8): replacing every reward $R_{t+k+1}$ with $R_{t+k+1}+c$ gives a new return

$$G_t' = \sum_{k=0}^{\infty}\gamma^k(R_{t+k+1}+c) = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1} + \sum_{k=0}^{\infty}\gamma^k c = G_t + c\sum_{k=0}^{\infty}\gamma^k$$

The second sum is a geometric series (as in Exercise 3.9/3.10), so for $\gamma<1$:

$$v_c = c\sum_{k=0}^{\infty}\gamma^k = \frac{c}{1-\gamma}$$

So $G_t' = G_t + v_c$ for every state and every policy — the shift is identical everywhere, so it adds the same constant to $v_\pi(s)$ for every $s$ under every $\pi$, leaving all relative comparisons between states/policies unaffected.

Exercise 3.16 Now consider adding a constant c to all the rewards in an episodic task,
such as maze running. Would this have any e↵ect, or would it leave the task unchanged
as in the continuing task above? Why or why not? Give an example.

[Ans.] Unlike the continuing case, this does have an effect — the sign of $c$ matters here.

In the continuing case, the constant contribution to the return was $v_c = c\sum_{k=0}^{\infty}\gamma^k = c/(1-\gamma)$, the *same* value regardless of policy, since every policy's return sums to infinity. In the episodic case the sum instead stops at the episode's termination time $T$:

$$G_0' = G_0 + c\sum_{k=0}^{T-1}\gamma^k$$

and this constant contribution now depends on $T$ — the episode length — which varies from one policy/trajectory to another. So the shift is no longer uniform across policies, and it can change which policy looks best.

Concretely, in the maze example, rewards are $0$ per step and $+1$ upon escaping. Adding $c>0$ to every reward turns this into $c$ per step and $1+c$ upon escaping — so every extra time step spent wandering before escaping now earns an additional $+c$. This gives the agent an incentive to *prolong* the episode rather than escape quickly (or, in the undiscounted case, to never escape at all, accumulating $c$ indefinitely), which directly undermines the original goal of finding the exit as fast as possible.

**[IMPORTANT]** Exercise 3.17 What is the Bellman equation for action values, that
is, for q⇡? It must give the action value q⇡(s, a) in terms of the action
values, q⇡(s0
,a0
), of possible successors to the state–action pair (s, a).
Hint: The backup diagram to the right corresponds to this equation.
Show the sequence of equations analogous to (3.14), but for action
values.

[Ans.] Combine the 3.13 result (which still has $v_\pi(s')$ inside it) with the 3.12 result (rewritten for state $s'$, with a fresh action variable $a'$):

$$q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma\,v_\pi(s')\big]$$

$$v_\pi(s') = \sum_{a'} \pi(a'\mid s')\, q_\pi(s',a')$$

Substituting the second into the first eliminates $v_\pi$ entirely, giving $q_\pi(s,a)$ purely in terms of the action values of its successors $(s',a')$:

$$q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma\sum_{a'}\pi(a'\mid s')\, q_\pi(s',a')\right]$$



**[IMPORTANT]** Exercise 3.18 The value of a state depends on the values of the actions possible in that
state and on how likely each action is to be taken under the current policy. We can
think of this in terms of a small backup diagram rooted at the state and considering each
possible action:
s
taken with
probability ⇡(a|s)
v⇡(s)
q⇡(s, a)
a1 a2 a3
Give the equation corresponding to this intuition and diagram for the value at the root

node, v⇡(s), in terms of the value at the expected leaf node, q⇡(s, a), given St = s. This
equation should include an expectation conditioned on following the policy, ⇡. Then give
a second equation in which the expected value is written out explicitly in terms of ⇡(a|s)
such that no expected value notation appears in the equation. ⇤

[Ans.] The root's value is the average of the leaf values $q_\pi(s,a)$, where the averaging is over which action $A_t$ the policy randomly picks:

$$v_\pi(s) = \mathbb{E}_\pi\big[q_\pi(s, A_t) \mid S_t=s\big]$$

Writing that expectation out explicitly, weighting each leaf $q_\pi(s,a)$ by the probability $\pi(a\mid s)$ of the policy choosing that branch:

$$v_\pi(s) = \sum_a \pi(a\mid s)\, q_\pi(s,a)$$


**[IMPORTANT]** Exercise 3.19 The value of an action, q⇡(s, a), depends on the expected next reward and
the expected sum of the remaining rewards. Again we can think of this in terms of a
small backup diagram, this one rooted at an action (state–action pair) and branching to
the possible next states:
s, a q⇡(s, a)
s0
3 s0
2 s0
1
r1 r2 r3 v⇡(s0
)
expected
rewards
Give the equation corresponding to this intuition and diagram for the action value,
q⇡(s, a), in terms of the expected next reward, Rt+1, and the expected next state value,
v⇡(St+1), given that St =s and At =a. This equation should include an expectation but
not one conditioned on following the policy. Then give a second equation, writing out the
expected value explicitly in terms of p(s0
,r|s, a) defined by (3.2), such that no expected
value notation appears in the equation.

[Ans.] With $S_t=s$ and $A_t=a$ both already fixed, the only remaining randomness is the environment's response — no $\pi$ needed:

$$q_\pi(s,a) = \mathbb{E}\big[R_{t+1} + \gamma\, v_\pi(S_{t+1}) \mid S_t=s, A_t=a\big]$$

($v_\pi(S_{t+1})$ replaces $G_{t+1}$ here by the law of iterated expectations: $\mathbb{E}[G_{t+1}\mid S_{t+1}=s']=v_\pi(s')$ from Exercise 3.13, so averaging $v_\pi(S_{t+1})$ over $S_{t+1}$ gives the same result as averaging $G_{t+1}$ directly.)

Writing the expectation out explicitly over the four-argument $p$ (same expansion as Exercise 3.13):

$$q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_\pi(s')\big]$$

Exercise 3.20 Draw or describe the optimal state-value function for the golf example. ⇤

[Ans.] $v_{putt}(s)$ (the value of always putting) has contours near the hole labeled $-1, -2, -3,\dots$, growing outward, with a very deep dip over the sand trap since escaping it by putting alone takes many strokes. $v_*(s)$ matches $v_{putt}(s)$ exactly within putting range of the hole, since putter is already optimal there. Everywhere farther out, $v_*(s) \geq v_{putt}(s)$ and generally strictly greater: the driver covers far more distance per stroke, so locations that would take 3+ putts to hole out can be reached in 2 strokes (drive, then putt) under the optimal policy. So the $-2$ contour of $v_*$ extends much farther from the hole than the $-2$ contour of $v_{putt}$. The sand trap is still a locally low-value region under $v_*$ (an extra stroke is still needed to escape it), just less catastrophic than under $v_{putt}$.

Exercise 3.21 Draw or describe the contours of the optimal action-value function for
putting, q⇤(s, putter), for the golf example. ⇤

[Ans.] $q_*(s,\text{putter})$ equals $v_{putt}(s)$ (and equals $v_*(s)$) within putting range, since committing to the putter there is already optimal. Outside putting range, $q_*(s,\text{putter})$ is the value of being *forced* to putt once from $s$ (a short move), then playing optimally afterward (switching to the driver if useful). This sits between the other two: worse than $v_*(s)$ (which would use the driver immediately, from wherever is genuinely optimal), but better than $v_{putt}(s)$ (which forces putting for every remaining stroke, not just the first). So its contours look like $v_{putt}(s)$'s contours shifted outward by roughly the distance covered in one putt, since after that first forced putt the agent recovers optimal play.
0 +2 +1 0
left right
Exercise 3.22 Consider the continuing MDP shown to the
right. The only decision to be made is that in the top state,
where two actions are available, left and right. The numbers
show the rewards that are received deterministically after
each action. There are exactly two deterministic policies,
⇡left and ⇡right. What policy is optimal if  = 0? If  =0.9?
If  =0.5?

[Ans.] Each policy sends the agent around a length-2 cycle back to the top state: $\pi_{left}$ gives rewards $1,0,1,0,\dots$; $\pi_{right}$ gives rewards $0,2,0,2,\dots$. Since each reward reappears every 2 steps, both values are geometric series in $\gamma^2$:

$$v_{left}(\text{top}) = 1+\gamma(0)+\gamma^2(1)+\dots = 1\cdot(1+\gamma^2+\gamma^4+\dots) = \frac{1}{1-\gamma^2}$$

$$v_{right}(\text{top}) = 0+\gamma(2)+\gamma^2(0)+\dots = 2\gamma\cdot(1+\gamma^2+\gamma^4+\dots) = \frac{2\gamma}{1-\gamma^2}$$

Both share the same positive denominator, so comparing them reduces to comparing $1$ vs. $2\gamma$: right is optimal when $2\gamma>1 \iff \gamma>0.5$, left when $\gamma<0.5$, and they're tied at $\gamma=0.5$ (both deterministic policies achieve the same value).

- $\gamma=0$: $2\gamma=0<1 \Rightarrow \pi_{left}$ optimal.
- $\gamma=0.9$: $2\gamma=1.8>1 \Rightarrow \pi_{right}$ optimal.
- $\gamma=0.5$: $2\gamma=1 \Rightarrow$ tied, both optimal.

Exercise 3.23 Give the Bellman equation for q⇤ for the recycling robot. ⇤

[Ans.] Applying $q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)[r+\gamma\max_{a'} q_*(s',a')]$ to each state-action pair, using the transition table from Exercise 3.4:

$$q_*(\text{high,search}) = \alpha\big[r_{search}+\gamma\max_{a'}q_*(\text{high},a')\big] + (1-\alpha)\big[r_{search}+\gamma\max_{a'}q_*(\text{low},a')\big]$$

$$q_*(\text{high,wait}) = r_{wait} + \gamma\max_{a'}q_*(\text{high},a')$$

$$q_*(\text{low,search}) = (1-\beta)\big[{-3}+\gamma\max_{a'}q_*(\text{high},a')\big] + \beta\big[r_{search}+\gamma\max_{a'}q_*(\text{low},a')\big]$$

$$q_*(\text{low,wait}) = r_{wait} + \gamma\max_{a'}q_*(\text{low},a')$$

$$q_*(\text{low,recharge}) = \gamma\max_{a'}q_*(\text{high},a')$$

Exercise 3.24 Figure 3.5 gives the optimal value of the best state of the gridworld as
24.4, to one decimal place. Use your knowledge of the optimal policy and (3.8) to express
this value symbolically, and then to compute it to three decimal places. ⇤

[Ans.] The best state is $A$. The optimal policy jumps immediately from $A$ to $A'$ (reward $+10$), then takes the shortest path back to $A$ — 4 steps, each with reward $0$ — before jumping again. So reward $+10$ recurs every 5 steps, giving a repeating geometric pattern:

$$v_*(A) = \sum_{k=0}^{\infty} \gamma^{5k}\cdot 10 = \frac{10}{1-\gamma^5}$$

With $\gamma=0.9$: $\gamma^5 = 0.9^5 = 0.59049$, so

$$v_*(A) = \frac{10}{1-0.59049} = \frac{10}{0.40951} \approx 24.419$$

which matches the figure's $24.4$ (to one decimal place).
Exercise 3.25 Give an equation for v⇤ in terms of q⇤. ⇤

[Ans.] The optimal policy is greedy — it puts all its probability on whichever action maximizes $q_*(s,a)$, so the weighted average from Exercise 3.12 ($v_\pi(s)=\sum_a \pi(a\mid s)q_\pi(s,a)$) collapses to a simple max, with no $\pi$ involved:

$$v_*(s) = \max_a q_*(s,a)$$

Exercise 3.26 Give an equation for q⇤ in terms of v⇤ and the four-argument p. ⇤

[Ans.] Same structure as Exercise 3.13, with $*$ in place of $\pi$ — this relationship only involves the environment's dynamics $p$, not any particular policy:

$$q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma\, v_*(s')\big]$$

Exercise 3.27 Give an equation for ⇡⇤ in terms of q⇤. ⇤
Exercise 3.28 Give an equation for ⇡⇤ in terms of v⇤ and the four-argument p. ⇤
Exercise 3.29 Rewrite the four Bellman equations for the four value functions (v⇡, v⇤, q⇡,
and q⇤) in terms of the three argument function p (3.4) and the two-argument function r
(3.5).

### Notes while reading Chapter 3 

reward hypothesis:
That all of what we mean by goals and purposes can be well thought of as
the maximization of the expected value of the cumulative sum of a received
scalar signal (called reward). (pg 53)

The reward signal is your way of communicating to
the agent what you want achieved, not how you want it achieved. (pg 54)

## Chapter 4 Dynamic Programming 

zDP algorithms are obtained by
turning Bellman equations such as these into assignments, that is, into update rules for
improving approximations of the desired value functions.

Exercise 4.1 In Example 4.1, if ⇡ is the equiprobable random policy, what is q⇡(11, down)?
What is q⇡(7, down)?

[Ans.] Every transition is deterministic here and $\gamma=1$, so $q_\pi(s,a) = r + v_\pi(s')$ where $s'$ is wherever action $a$ leads.

$$q_\pi(11,\texttt{down}) = -1 + v_\pi(\text{terminal}) = -1 + 0 = -1$$

$$q_\pi(7,\texttt{down}) = -1 + v_\pi(11) = -1 + (-14) = -15$$

Exercise 4.2 In Example 4.1, suppose a new state 15 is added to the gridworld just below
state 13, and its actions, left, up, right, and down, take the agent to states 12, 13, 14,
and 15, respectively. Assume that the transitions from the original states are unchanged.
What, then, is v⇡(15) for the equiprobable random policy? Now suppose the dynamics of
state 13 are also changed, such that action down from state 13 takes the agent to the new
state 15. What is v⇡(15) for the equiprobable random policy in this case? ⇤

[Ans.] **First part:** state 15's actions are left$\to12$, up$\to13$, right$\to14$, down$\to$itself, and the original states are unchanged, so $v(12)=-22,v(13)=-20,v(14)=-14$ still hold. Letting $x=v_\pi(15)$:

$$x = \frac{1}{4}\big[(-1-22)+(-1-20)+(-1-14)+(-1+x)\big] = \frac{1}{4}(-60+x) \Rightarrow 3x=-60 \Rightarrow x=-20$$

**Second part:** now state 13's `down` also goes to 15 instead of looping back to itself, so $v_\pi(13)$ and $v_\pi(15)$ depend on each other. Using unchanged $v(9)=-20,v(12)=-22,v(14)=-14$, and letting $x=v_\pi(13)$, $y=v_\pi(15)$:

$$x = \frac{1}{4}\big[(-1-22)+(-1-14)+(-1-20)+(-1+y)\big] = \frac{1}{4}(-60+y) \Rightarrow y=4x+60$$

$$y = \frac{1}{4}\big[(-1-22)+(-1+x)+(-1-14)+(-1+y)\big] = \frac{1}{4}(-40+x+y) \Rightarrow x=3y+40$$

Solving simultaneously: $x=3(4x+60)+40=12x+220 \Rightarrow x=-20$, then $y=4(-20)+60=-20$. So $v_\pi(13)=-20$ (unchanged) and $v_\pi(15)=-20$ — the same value as the first part.

Exercise 4.3 What are the equations analogous to (4.3), (4.4), and (4.5), but for actionvalue functions instead of state-value functions?

[Ans.] Mirroring each of (4.3)-(4.5) for $q_\pi$ instead of $v_\pi$:

**(4.3)-analog** — compact expectation form, from the return recursion (same as Exercise 3.13's derivation):

$$q_\pi(s,a) \doteq \mathbb{E}_\pi\big[R_{t+1}+\gamma\, q_\pi(S_{t+1},A_{t+1}) \mid S_t=s, A_t=a\big]$$

**(4.4)-analog** — expanded explicitly via $p$ and $\pi$ (this is exactly the Exercise 3.17 result):

$$q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\Big[r+\gamma\sum_{a'}\pi(a'\mid s')\,q_\pi(s',a')\Big]$$

**(4.5)-analog** — turning the equation into an iterative update rule, replacing $q_\pi$ on the right with a current estimate $q_k$ to produce an improved estimate $q_{k+1}$ (the action-value version of iterative policy evaluation):

$$q_{k+1}(s,a) \doteq \sum_{s',r} p(s',r\mid s,a)\Big[r+\gamma\sum_{a'}\pi(a'\mid s')\,q_k(s',a')\Big]$$

Exercise 4.4 The policy iteration algorithm on page 80 has a subtle bug in that it may
never terminate if the policy continually switches between two or more policies that are
equally good. This is okay for pedagogy, but not for actual use. Modify the pseudocode
so that convergence is guaranteed. ⇤

Ans. 

Exercise 4.5 How would policy iteration be defined for action values? Give a complete
algorithm for computing q⇤, analogous to that on page 80 for computing v⇤. Please pay
special attention to this exercise, because the ideas involved will be used throughout the
rest of the book. ⇤


Exercise 4.6 Suppose you are restricted to considering only policies that are "-soft,
meaning that the probability of selecting each action in each state, s, is at least "/|A(s)|.
Describe qualitatively the changes that would be required in each of the steps 3, 2, and 1,
in that order, of the policy iteration algorithm for v⇤ on page 80. ⇤



Exercise 4.7 (programming) Write a program for policy iteration and re-solve Jack’s car
rental problem with the following changes. One of Jack’s employees at the first location
rides a bus home each night and lives near the second location. She is happy to shuttle
one car to the second location for free. Each additional car still costs $2, as do all cars
moved in the other direction. In addition, Jack has limited parking space at each location.
If more than 10 cars are kept overnight at a location (after any moving of cars), then an
additional cost of $4 must be incurred to use a second parking lot (independent of how
many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often
occur in real problems and cannot easily be handled by optimization methods other than
dynamic programming. To check your program, first replicate the results given for the
original problem.


Exercise 4.8 Why does the optimal
policy for the gambler’s problem have such a curious form? In particular, for capital of 50
it bets it all on one flip, but for capital of 51 it does not. Why is this a good policy? ⇤



Exercise 4.9 (programming) Implement value iteration for the gambler’s problem and
solve it for ph =0.25 and ph =0.55. In programming, you may find it convenient to
introduce two dummy states corresponding to termination with capital of 0 and 100,
giving them values of 0 and 1 respectively. Show your results graphically, as in Figure 4.3.
Are your results stable as ✓ ! 0? ⇤



Exercise 4.10 What is the analog of the value iteration update (4.10) for action values,
qk+1(s, a)?


Note -> chapter 4 dont make a lot of sennse especisally page 83 (of the book) -->