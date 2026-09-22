<!-- ---
layout: blog
title: "Transformers Laid Out"
date: 2025-01-3 12:00:00 +0530
categories: [personal, technology]
image: /assets/transformers_laid_out/meme.png
---

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




Exercise 2.9 Show that in the case of two actions, the soft-max distribution is the same
as that given by the logistic, or sigmoid, function often used in statistics and artificial
neural networks.


Exercise 2.10 Suppose you face a 2-armed bandit task whose true action values change
randomly from time step to time step. Specifically, suppose that, for any time step,
the true values of actions 1 and 2 are respectively 10 and 20 with probability 0.5 (case
A), and 90 and 80 with probability 0.5 (case B). If you are not able to tell which case
you face at any step, what is the best expected reward you can achieve and how should
you behave to achieve it? Now suppose that on each step you are told whether you are
facing case A or case B (although you still don’t know the true action values). This is an
associative search task. What is the best expected reward you can achieve in this task,
and how should you behave to achieve it?


'Exercise 2.11 (programming) Make a figure analogous to Figure 2.6 for the nonstationary
case outlined in Exercise 2.5. Include the constant-step-size "-greedy algorithm with
↵=0.1. Use runs of 200,000 steps and, as a performance measure for each algorithm and
parameter setting, use the average reward over the last 100,000 steps.


## Chapter 3 Finite Markov Decision Processes

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


Exercise 3.4 Give a table analogous to that in Example 3.3, but for p(s0
,r|s, a). It
should have columns for s, a, s0
, r, and p(s0
,r|s, a), and a row for every 4-tuple for which
p(s0
,r|s, a) > 0.

Exercise 3.5 The equations in Section 3.1 are for the continuing case and need to be
modified (very slightly) to apply to episodic tasks. Show that you know the modifications
needed by giving the modified version of (3.3).

Exercise 3.6 Suppose you treated pole-balancing as an episodic task but also used
discounting, with all rewards zero except for 1 upon failure. What then would the
return be at each time? How does this return di↵er from that in the discounted, continuing
formulation of this task? ⇤

[Ans.] 

Exercise 3.7 Imagine that you are designing a robot to run a maze. You decide to give it a
reward of +1 for escaping from the maze and a reward of zero at all other times. The task
seems to break down naturally into episodes—the successive runs through the maze—so
you decide to treat it as an episodic task, where the goal is to maximize expected total
reward (3.7). After running the learning agent for a while, you find that it is showing
no improvement in escaping from the maze. What is going wrong? Have you e↵ectively
communicated to the agent what you want it to achieve? ⇤

[Ans.] No, because the agent has no metric of knowing if it has improved over time. Nor any other incentive to do so.

Exercise 3.8 Suppose  =0.5 and the following sequence of rewards is received R1 = 1,
R2 = 2, R3 = 6, R4 = 3, and R5 = 2, with T = 5. What are G0, G1, ..., G5? Hint:
Work backwards. ⇤


Exercise 3.9 Suppose  =0.9 and the reward sequence is R1 = 2 followed by an infinite
sequence of 7s. What are G1 and G0? ⇤


Exercise 3.10 Prove the second equality in (3.10).
Sum of GP

Exercise 3.11 If the current state is St, and actions are selected according to a stochastic
policy ⇡, then what is the expectation of Rt+1 in terms of ⇡ and the four-argument
function p (3.2)?

Exercise 3.12 Give an equation for v⇡ in terms of q⇡ and ⇡. ⇤
Exercise 3.13 Give an equation for q⇡ in terms of v⇡ and the four-argument p.

Exercise 3.14 The Bellman equation (3.14) must hold for each state for the value function
v⇡ shown in Figure 3.2 (right) of Example 3.5. Show numerically that this equation holds
for the center state, valued at +0.7, with respect to its four neighboring states, valued at
+2.3, +0.4, 0.4, and +0.7. (These numbers are accurate only to one decimal place.) ⇤

Exercise 3.15 In the gridworld example, rewards are positive for goals, negative for
running into the edge of the world, and zero the rest of the time. Are the signs of these
rewards important, or only the intervals between them? Prove, using (3.8), that adding a
constant c to all the rewards adds a constant, vc, to the values of all states, and thus
does not a↵ect the relative values of any states under any policies. What is vc in terms
of c and ? ⇤


Exercise 3.16 Now consider adding a constant c to all the rewards in an episodic task,
such as maze running. Would this have any e↵ect, or would it leave the task unchanged
as in the continuing task above? Why or why not? Give an example.

Exercise 3.17 What is the Bellman equation for action values, that
is, for q⇡? It must give the action value q⇡(s, a) in terms of the action
values, q⇡(s0
,a0
), of possible successors to the state–action pair (s, a).
Hint: The backup diagram to the right corresponds to this equation.
Show the sequence of equations analogous to (3.14), but for action
values.

Exercise 3.18 The value of a state depends on the values of the actions possible in that
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


Exercise 3.19 The value of an action, q⇡(s, a), depends on the expected next reward and
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

Exercise 3.20 Draw or describe the optimal state-value function for the golf example. ⇤
Exercise 3.21 Draw or describe the contours of the optimal action-value function for
putting, q⇤(s, putter), for the golf example. ⇤
0 +2 +1 0
left right
Exercise 3.22 Consider the continuing MDP shown to the
right. The only decision to be made is that in the top state,
where two actions are available, left and right. The numbers
show the rewards that are received deterministically after
each action. There are exactly two deterministic policies,
⇡left and ⇡right. What policy is optimal if  = 0? If  =0.9?
If  =0.5?

Exercise 3.23 Give the Bellman equation for q⇤ for the recycling robot. ⇤
Exercise 3.24 Figure 3.5 gives the optimal value of the best state of the gridworld as
24.4, to one decimal place. Use your knowledge of the optimal policy and (3.8) to express
this value symbolically, and then to compute it to three decimal places. ⇤
Exercise 3.25 Give an equation for v⇤ in terms of q⇤. ⇤
Exercise 3.26 Give an equation for q⇤ in terms of v⇤ and the four-argument p. ⇤
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
 -->
