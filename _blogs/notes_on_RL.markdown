---
layout: blog
title: "Notes on RL"
date: 2026-10-01 12:00:00 +0530
categories: [personal, technology]
---

Now I am aware you must have been looking forward to the second part of my [CUDA blog](/blogs/supe_fast_inference/) (if you haven't read it yet, check it out!), but hey! A man can have varied interests. And I believe if you are trying to be a great ML engineer or developer, or are just EXTREMELY enthusiastic about the space, Reinforcement Learning must have tickled your brain as well.

In my opinion, if AI is magic to Computer Science, RL is magic to AI.

The usual school of thought while talking about RL (or any other ML topic for the most part) is to first lay out a table of contents, show what will be covered, what the problems are, yada yada yada. We are gonna do none of that. We are innovative people and we laugh in the face of the old ways.

So we will do what innovators do: we will think of the simplest problem we can, make a few assumptions, try to solve it, and slowly make it more complex.

I invite you to read the following work with an OPEN MIND. So let us begin by first framing a problem.

## The first problem

Let's say you did some odd job for your neighbour and your naive lil self just got your first paycheck!

![Our hero receiving their first paycheck](/assets/blog_assets/notes_on_RL/notes_on_rl_1.webp)

Now as you are walking down the street, you find this amazing place called a "Casino" and they tell you that you can double your money here. So you walk in...

![The casino: double your money!](/assets/blog_assets/notes_on_RL/notes_on_rl_2.webp)

Oh god, what is this ungodly place! You are startled by all the bright lights, the money flying around, the vomit-colored carpet. But you are filled with joy, because you are about to double your money!!!

![Inside the casino](/assets/blog_assets/notes_on_RL/notes_on_rl_3.webp)

As you are walking through this labyrinth, you discover a fairly simple-looking machine. Well, a bunch of them in fact, lined up one after the other. They are slot machines!

![Three slot machines](/assets/blog_assets/notes_on_RL/notes_on_rl_4.webp)

You think maybe you should try your luck here, as they seem simpler than poker: you just need to put money in, and get money out.

You try the first machine (let's assume we put in a dollar and if we win we get an unspecified amount of money back; if we lose, THE MACHINE EATS OUR MONEY!!!). After 30 tries you find that instead of having more money, you have lost a significant amount.

This cannot be right, the hoarding said that the house never cheats (oh you naive kid, if only the world was as innocent as you are), and that you will in fact double your money. Sure, you will lose some tries, but if the claim is true, then on average, if you play enough times, you should win more than you lose!

So you start thinking and looking around, and that's when you observe that the man on machine number 3 seems to be winning quite a fair bit. So you wait for him to leave, and once he does, you go to that machine and try it 30 times. Lo and behold... you have made more money than you started with! That's when you realise... "THE HOUSE DOES IN FACT CHEAT!" (Who would have guessed, right?) Now, you are an intrepid person, who decides to fight back against this indignity with math and statistics.

So you formulate how you can win more money.

Let us assume we have $N$ tries (the amount of money), and we have $k$ options in front of us (the number of slot machines). We can assume that each of these $k$ options has an expected return, i.e. a mean around which there is some variance, but if played enough times, the average of what it gives back will converge to its true value. (This is the [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers): given enough tries, the black box will give its average output.)

So let's assume this perfect, actual value of a slot machine $a$ can be represented as $q_\ast(a)$. But the problem is, any time we use it, it does not return the perfect $q_\ast(a)$ (because if it did, everyone would play all the slot machines once, figure out which gives the highest payout and just use that! So instead they follow a distribution with some variance and a mean). So instead we keep a running estimate $Q_n(a)$, which is the average of the rewards we have received from that machine so far.

We can write the true value as

$$
q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]
$$

i.e. the expected reward $R_t$ given that we took the action $A_t = a$ (here an action is you choosing a particular slot machine).

> If this is your first time seeing $\mathbb{E}[X]$, it essentially is the weighted mean of a distribution. In simpler terms it can be written as
>
> $$\mathbb{E}[X] = \sum_i x_i \, p(x_i)$$
>
> i.e. the value of any given $x$ multiplied by the probability of that $x$ appearing, all added up. (Now if it is a uniform distribution, the probability of any given value occurring is $\frac{1}{\text{number of values}}$, so for something like the expected value of a die it would be
>
> $$\mathbb{E}[\text{die}] = 1 \cdot \tfrac{1}{6} + 2 \cdot \tfrac{1}{6} + 3 \cdot \tfrac{1}{6} + 4 \cdot \tfrac{1}{6} + 5 \cdot \tfrac{1}{6} + 6 \cdot \tfrac{1}{6} = \frac{21}{6} = 3.5$$
>
> Notice that you can never actually roll a $3.5$! The expected value is not "the most likely outcome", it is the average you would get if you rolled the die a huge number of times, which is exactly the Law of Large Numbers from above.)

> If this is your first time seeing the notation $\mathbb{E}[X \mid Y]$, it comes from conditional probability. $P(A \mid B)$ means: given that $B$ happened, what is the probability that $A$ also happened? I like to imagine this using Venn diagrams. Once we know $B$ happened, $B$ becomes our whole world, and we ask how much of that world is also $A$:
>
> $$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$
>
![Venn diagram of conditional probability](/assets/blog_assets/notes_on_RL/notes_on_rl_13.webp)
>
> The conditional *expectation* $\mathbb{E}[X \mid Y = y]$ uses this same idea, but instead of a probability it gives you an average: given that $y$ happened, what is the average value of $X$? It is just the weighted mean from above, with the conditional probabilities as the weights: $\mathbb{E}[X \mid Y = y] = \sum_x x \, P(X = x \mid Y = y)$. So $\mathbb{E}[R_t \mid A_t = a]$ reads "the average reward, given that we picked machine $a$".

> A further explanation of the above idea comes from introducing the ideas of posterior, prior, likelihood and marginal probability. While these sound like super complex words, they are quite simple to understand. We can write [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) as
>
> $$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$
>
> Here $P(A \mid B)$ is the **posterior**, essentially the thing we are trying to find out (our belief about $A$ *after* seeing $B$). $P(A)$ is the **prior**, what we already believed about $A$ *before* seeing anything. $P(B \mid A)$ is the **likelihood**, how likely the evidence $B$ would be if $A$ were true. And $P(B)$ is called the **marginal probability** (or evidence), the overall probability of seeing $B$ at all.

We can write our estimate of a machine after it has been played $n-1$ times as

$$
Q_n = \frac{R_1 + R_2 + \cdots + R_{n-1}}{n-1}
$$

This can be simplified (so that we do not need to store every single reward we have ever received). The estimate after $n$ rewards is

$$
\begin{aligned}
Q_{n+1} &= \frac{1}{n}\sum_{i=1}^{n} R_i \\
&= \frac{1}{n}\left(R_n + \sum_{i=1}^{n-1} R_i\right) \\
&= \frac{1}{n}\left(R_n + (n-1)\frac{1}{n-1}\sum_{i=1}^{n-1} R_i\right) \\
&= \frac{1}{n}\big(R_n + (n-1)Q_n\big) \\
&= \frac{1}{n}\big(R_n + nQ_n - Q_n\big) \\
&= Q_n + \frac{1}{n}\big[R_n - Q_n\big]
\end{aligned}
$$

In the third line we multiplied and divided by $(n-1)$, which lets us spot that $\frac{1}{n-1}\sum_{i=1}^{n-1} R_i$ is just our old estimate $Q_n$. So now for each machine we only need to remember two numbers: the current estimate $Q_n$ and the count $n$.

> This is a common trick in a lot of machine learning and you will see it in many papers. It is so common, in fact, that they often skip this exact derivation.

All in all, we can estimate the expected value of any slot machine simply by

$$
\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize}\big[\text{Reward} - \text{OldEstimate}\big]
$$

where the step size here is $\frac{1}{n}$.

You try this with the 3 machines, find the one which gave you the highest expected reward, make a huge buck and leave for home happy.
You come back the next day, only to realise the casino caught on to what you were doing. So instead of 3 slot machines, they now have 10!!! machines.

![The casino added more machines](/assets/blog_assets/notes_on_RL/notes_on_rl_5.webp)

Your previous method will not work anymore, because you will waste a lot of tries just trying to find the optimal machine.

So you pull out your trusty notebook and start thinking:

"What if I assume that every machine gives on average 0 returns, then I try a machine and keep that estimate. Now that is my highest returning machine at the moment, so I will keep exploiting it, and at random times (let's say $\varepsilon$ of the time) I will explore and try a random machine. If that gives me a greater reward than my current estimate for my current machine, I will stick to that!"

Wow, you mad genius. You write down your formula as such (mad geniuses need algorithms to work for some reason):

> Note: This method is formally called *$\varepsilon$-greedy* action selection, and the dilemma here is between **exploitation** (just using the machine that has given you the highest average so far) and **exploration** (trying other machines which could potentially have a higher average). We vary the value of $\varepsilon$ to figure out what works best for us!

```python
import numpy as np

rng = np.random.default_rng()

def bandit(q_true, action):
    # the slot machine: pays out a noisy reward around its (hidden) true value
    return rng.normal(loc=q_true[action], scale=1.0)

def epsilon_greedy(q_true, epsilon=0.1, steps=1000, initial_value=0.0):
    k = len(q_true)
    Q = np.full(k, initial_value)  # our estimate of each machine
    N = np.zeros(k)                # how many times we have played each machine
    rewards = np.zeros(steps)

    for t in range(steps):
        if rng.random() < epsilon:
            A = rng.integers(k)                           # explore
        else:
            A = rng.choice(np.flatnonzero(Q == Q.max()))  # exploit, breaking ties randomly
        R = bandit(q_true, A)
        N[A] += 1
        Q[A] += (1 / N[A]) * (R - Q[A])
        rewards[t] = R

    return Q, rewards

q_true = rng.normal(0, 1, size=10)  # 10 machines, each with a hidden true value
Q, rewards = epsilon_greedy(q_true, epsilon=0.1)
print("best machine:", q_true.argmax(), "| our best guess:", Q.argmax())
print("average reward:", rewards.mean())
```

> Now now, I know it is tempting to skip reading the code above, but just give it a glance. It is quite simple and essential to our understanding.

You keep doing this for a while, but you are not getting the returns you would like, mostly because you are stuck exploiting only a few machines, while there are many more which could potentially have much higher rewards.

So you put on your thinking cap again.

"What if I assume every machine gives me, instead of 0, a value of 10 as the initial reward? This way I will be incentivized to try every machine at least once!"

And you have done it again! How do you even do it?

So now you modify your algorithm by changing $Q(a) \leftarrow 10$ (in the code above, that is just `initial_value=10`).

> Formally this is called **optimistic initial values**, the idea is to force the agent to explore all options at least once! But it has a problem, as we will see soon...

You do this for a while, but again, you are distraught with the results. Because as you keep playing, you are also keeping track of how much money you are making and losing, and the graph does not look as good as you would like it to. The obvious answer seems to be that even after you try all the machines, in the end you still get stuck with a few machines, because you have no incentive to explore.

So you realise,

"What if I give these a high initial value, try them all, and also keep track of how many times I have tried each machine? This way, if I have been exploiting a specific machine for too long, I can look at which machines I have tried the least and play them. As these are the machines I have used the least, they have the highest chance of having a wrong $Q$ estimate."

Oh my my my, are you [Edward O. Thorp](https://en.wikipedia.org/wiki/Edward_O._Thorp) by any chance? You are on fire!!!

You modify how you pick your action $A_t$ as

$$
A_t \doteq \arg\max_a \left[ Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

> If the above equation feels like a big jump, let me simplify it. $\arg\max_a$ essentially means "pick the $a$ (the argument) that makes the thing in the brackets the biggest". Here the thing in the brackets is our estimate *plus* an exploration bonus, so it is not just the highest expected return at that moment. $\ln$ is log with the natural base $e \approx 2.718$ ([Euler's number](https://en.wikipedia.org/wiki/E_(mathematical_constant)), read more about it here!). The way I like to remember log is something like the following. Imagine we take log with base 10, we can write $\log_{10}(1000) = 3$, i.e. how many times do we need to multiply 10 by itself to get 1000? Or, $10^x = 1000$. Obviously I took a very easy number because it shows the idea. The main reason we use log is because it scales the value much better: it keeps growing, but slower and slower. Compare the two graphs below for yourself to understand what I mean!
>
> ![Linear vs log curve](/assets/blog_assets/notes_on_RL/notes_on_rl_12.webp)
>
> After 1000 pulls, $t$ is 1000 but $\ln t$ is only about 6.9. So the exploration bonus keeps nudging us to revisit neglected machines, but it never grows so fast that it drowns out what we have actually learned in $Q_t(a)$.

$N_t(a)$ is the number of times you have pulled the lever of a specific machine, $t$ is the total number of pulls so far (and $\ln t$ is its natural log, so the bonus grows slowly over time), and $c > 0$ controls how much you care about exploring. This will give a greater bonus to the machines you have tried the least, and a machine you have not tried at all ($N_t(a) = 0$) is treated as the best choice! (Now now, you are a smart person, think for a minute and realise this is simpler than it looks!)

> NOTE: This is *Upper Confidence Bound (UCB)* action selection. The square-root term is the 'uncertainty' in the estimate. It shrinks as you play a machine more ($N_t(a)$ grows) and slowly grows for machines you ignore (because $\ln t$ keeps growing).

You employ this method, get an absurd amount of money and go home.

You come back the next day, again to milk these losers.

You start playing... and after a while you realise... the mean of all the machines keeps changing over time. YOU ARE SHOCKED!

"Are these guys changing the expected return of a machine over time?"

Guess what, the casino caught on to your tricks again, and they changed the machines overnight to have a moving expected mean.

You are in a dismal state, you have lost faith, time and money. You think of giving up. That is when you think of Papa John, and realise he would never have given up on making pizza to feed his family. So you... take out your notebook one more time, to teach this casino that you are better than them.

You look at your formula and realise the fatal flaw is in the step multiplier. With $\frac{1}{n}$, every new reward matters less and less as $n$ grows, which only makes sense if the values converge to one value. But now these values are never going to converge to one value. So you decide to replace that with a constant $\alpha$, so that recent rewards always count more than old ones!

$$
Q_{n+1} = Q_n + \alpha\big[R_n - Q_n\big], \qquad \alpha \in (0, 1]
$$

This small change changes everything. Now you are back on track and winning some more!

> Note: In RL terms, a problem where the true values keep changing over time is called a *nonstationary* problem. And the whole slot machine setup, where there is only one situation and you just keep picking an action, is called a *non-associative* problem, or the *$k$-armed bandit* problem.

The casino has had it with you and your math! The manager sends goons towards you to chase you out! You run towards the back exit...

![The manager has had it with your maths](/assets/blog_assets/notes_on_RL/notes_on_rl_6.webp)

...and just when you thought you had escaped your hell and were a free man, you realise... the back gate led to a maze, with only two ends: a dangerous fire pit, or the gates of Saintsbury (this is what we want!). Our hero is again in peril.

Just as you are about to lose hope, you spot something pinned on the wall by the entrance... A MAP of the maze! It shows every corridor, every dead end, where each turn leads, and where the fire pit and the gates of Saintsbury are. (How convenient... almost too convenient. But you are in no position to complain.)

![The map of the maze](/assets/blog_assets/notes_on_RL/notes_on_rl_7.webp)

Our hero is quite exhausted from all the trouble and is in no position to solve this complex maze by hand.
(The maze only looks simple to us as spectators, but in reality it is far too complex to be solved in mere seconds!)

That is when you take out your trusty sci-fi robo partner, Maurice! Now you must write an algorithm to run Maurice on, so Maurice can find the gates of Saintsbury for you and you can escape this maze!

Okay, now designing an algo for Maurice is going to be an arduous task, so you start by first breaking down your variables.

![Agent, environment, actions and state](/assets/blog_assets/notes_on_RL/notes_on_rl_10.webp)

Maurice is your **agent**, who interacts with the **environment**, and where he is in the environment is his current **state**. Maurice can take **actions** (move up, down, left, right), and we would also like to give Maurice a **reward** if he gets the job done and saves us from this peril we are stuck in.

We can express the above idea in a simple diagram like below

![The agent-environment loop](/assets/blog_assets/notes_on_RL/notes_on_rl_8.webp)

We can also rationalize that we start in a state $S_0$ (like standing at the entrance of the maze), take an action $A_0$ (like moving forward), and because of that get a reward $R_1$ (in this case the reward will be 0, because it is not like we got out!) and end up in state $S_1$ (the next tile). From $S_1$ we take action $A_1$, get a reward $R_2$, and so on... till we reach the end (the terminal state, in our case either impending doom in the fire pit or heaven by escaping through the gates of Saintsbury):

$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots
$$

(Notice that the reward for the action taken at time $t$ is called $R_{t+1}$, because it arrives together with the next state $S_{t+1}$.)

The probability of ending up in state $s'$ with reward $r$, given that we were in state $s$ and took action $a$, can be written mathematically as

$$
p(s', r \mid s, a) \doteq \Pr\{S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a\}
$$

> Now now, this is nothing new, we already saw conditional probability in the beginning. This essentially says: given what is on the right is true (we are in a state and we took an action), what is the probability of the pair on the left (the state we will end up in and the reward we will get for it)!

And this is exactly what the map gives us! For every state and every action, we can read off where Maurice will end up and what reward he will get. (In a simple maze, each move takes you to exactly one next cell, so $p$ is just $1$ for that cell and $0$ for everything else. We write it as a probability so that it also works for trickier worlds, say a slippery floor that sometimes sends you somewhere else.)

Notice that $p$ only depends on where Maurice is *now* and what he does *now*, not on the whole path that got him there. This is called the **Markov property**, and a problem set up like this (states, actions, rewards and $p$) is called a **Markov Decision Process (MDP)**.

> Take this assumption for now, but later on we will expand more on MDPs, show how they work in most scenarios, and why they are the cornerstone of RL.

As you start formulating the problem, one of the first things that you realise is: the rewards are the easy part. Reaching the gates of Saintsbury gets $+1$, falling into the fire pit gets $-1$, and every other step gets $0$. But that is not enough! When Maurice is standing in some corridor in the middle of the maze, the reward there is $0$, which tells him nothing about whether he is one step from freedom or one step from the fire pit (the reason being... WE IN A MAZE! Every corridor looks the same!).

> Notice that the only way we tell Maurice what we want is through rewards. This idea is at the heart of RL and is called the **reward hypothesis**: all goals can be described as the maximization of the expected cumulative reward. The reward tells the agent *what* we want achieved, not *how* to achieve it. (We never tell Maurice "go left at the third corridor", we just reward him for getting out.)

What Maurice needs is not the reward of each state, but how *good* each state is in the long run, i.e. how much reward he can expect to collect from there onwards. We call this the **value** of a state. Rewards are what the maze hands out, values are what Maurice has to figure out.

> Let us slow down a bit if that felt like too much. Work backwards: if we get out, that gives us a reward. Our current problem is that we do not know how close we are to getting rewarded from any given state, so we need this idea of value. For instance, the value of the tile just before the gates is obviously higher than that of a tile 5 steps before it (once we add a little trick called *discounting* in a moment, which makes rewards that are further away count for less). And the value of the tile just before the fire pit is lower than that of the tiles which bring us closer to the gate.

The thing working in your favour is the map. Since we know the whole maze, Maurice does not need to take a single step to work these values out. He can sit right here and *think*. (Also I forgot to tell you, but he is essentially immortal, because you can respawn him every time he dies using your caller gadget, but let's not test that.)

So you think, okay, maybe I can initialize a value for each state, then use the map to keep updating how close each state gets me to the end goal.

(Woah, that was a hard sentence to say, let's break it down.)

Unlike the previous problem, where we got our reward immediately, here we get our reward after a while, so we can keep track of all the rewards within one run as

$$
G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T
$$

(the total reward you collect from time $t$ until the run ends at time $T$, by following whatever way of acting Maurice currently has. We call this the **return**, and we will talk about the optimal way of acting in a bit.)

But the problem with the above is that it can potentially explode (i.e. become numerically intractable, because the sum can get extremely large for large mazes, or even infinite if the task never ends). Another problem is that the agent will look far too much into the future, valuing every reward equally no matter how far away it is (this can lead Maurice to wander around collecting all the rewards, while what we want him to do is get us out ASAP). So what we can do is introduce an exponential weighting value $\gamma$ (with $0 \le \gamma \le 1$), called the **discount factor**:

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

> This is a good time to introduce a small idea called **episodic** and **continuing** tasks. The example we are dealing with now is episodic, i.e. it eventually ends. But there are multiple scenarios in real life where a task does not end. For instance, think of a thermostat trying to keep the temperature of a room constant. In all practicality it will never be done. Now you may wonder why we need to talk about the difference between episodic and continuing tasks. The big reason is that the math differs significantly between them. For instance, look at the above equation. It looks a lot like a [geometric progression](https://en.wikipedia.org/wiki/Geometric_series) (read more here), and the sum of an infinite GP is very different from a finite one. In an episodic task the sum stops at $T$, so it is always finite. In a continuing task it never stops, and without $\gamma$ it could blow up to infinity. Interestingly, for $0 \le \gamma < 1$, $\sum_{k=0}^{\infty} \gamma^k = \frac{1}{1-\gamma}$, a constant. So if every reward is at most $R_{\max}$, the return can never be bigger than $\frac{R_{\max}}{1-\gamma}$, finite no matter how long the task runs! (And as a bonus, if you add the same constant $c$ to every reward, every state's value just shifts by the same $\frac{c}{1-\gamma}$, so what really matters is the relative difference between rewards, not their actual values.) This part was more complex than I would have liked it to be, but as we move forward and do more RL, I will try to simplify it as we get more comfortable with this concept.

> There is another benefit of $\gamma$ for the infinite case: it essentially makes the task pseudo-episodic from any state. For $n$ large enough, $\gamma^n$ will be very close to zero, so every reward after that point contributes almost nothing. (A handy rule of thumb: the agent effectively looks about $\frac{1}{1-\gamma}$ steps ahead, e.g. around 10 steps for $\gamma = 0.9$.)

This is our discounted return. Now, using this discounted return, we can determine how valuable any current state is as

$$
v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\middle|\, S_t = s\right], \quad \text{for all } s \in \mathcal{S}
$$

Here $\pi$ is Maurice's **policy**, i.e. his way of behaving: $\pi(a \mid s)$ is the probability that Maurice picks action $a$ when he is in state $s$. How valuable a state is depends on how Maurice behaves from there on, which is why $v$ carries that little $\pi$.

You can think of the policy as Maurice's brain: it takes in the state and tells him what to do. It can be **deterministic**, written $a = \pi(s)$, always picking the same action in a given state. Or it can be **stochastic**, written $\pi(a \mid s)$, giving a probability for each action (right 70% of the time, left 20%, and so on).

![Deterministic vs stochastic policy](/assets/blog_assets/notes_on_RL/notes_on_rl_11.webp)

Finding the best policy, the one that collects the most reward (we call it $\pi_\ast$), is the whole goal of RL. There are two broad ways to get there:

- **Policy-based methods:** directly teach Maurice which action to take in each state.
- **Value-based methods:** teach Maurice how valuable each state is, then let him take the action that leads to the most valuable state.

![Policy based vs value based methods](/assets/blog_assets/notes_on_RL/notes_on_rl_9.webp)

Everything we do in this post is value-based. We will meet policy-based methods later in the series.

> The above explanation makes sense logically, but let's break it down mathematically as well. When we write $\mathbb{E}_\pi[X]$ we essentially "mean" (haha, pun intended) the expected value of the random variable $X$ when Maurice behaves according to $\pi$. (Quick short note: a [random variable](https://en.wikipedia.org/wiki/Random_variable) is an idea from probability and not the same thing as a variable in computer science. It is a quantity whose value depends on a random outcome, like the number a die lands on. Read more about it here.) Which we can break down as follows.
>
> The return $G_t$ is a random variable: every time Maurice starts from $s$, he can end up taking a different path and collecting a different total reward. *Which* paths are likely depends on two things: Maurice's choices ($\pi$) and how the maze responds ($p$). The little $\pi$ under the $\mathbb{E}$ is a reminder that the probabilities we average with come from following $\pi$. So using the weighted-mean definition from the beginning:
>
> $$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \sum_{g} g \cdot \Pr_\pi(G_t = g \mid S_t = s)$$
>
> i.e. every possible return $g$, weighted by how likely Maurice is to get it when starting from $s$ and following $\pi$. A tiny example: say from tile $s$, Maurice goes left half of the time ($\pi(\text{left} \mid s) = 0.5$), which always ends up at the gates with a return of $+1$, and goes right the other half, which always ends in the fire pit with a return of $-1$. Then $v_\pi(s) = 0.5 \cdot (+1) + 0.5 \cdot (-1) = 0$. Change his policy to go left 90% of the time, and the same tile is now worth $0.9 - 0.1 = 0.8$. Same tile, same maze, different policy, different value. That is why $v$ needs its $\pi$!

Now the problem with the above formulation is that we cannot really work with it, so we have to break it down into what we understand.

We can break it down as the following

$$
\begin{aligned}
v_\pi(s) &\doteq \mathbb{E}_\pi[G_t \mid S_t = s] && (1) \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s] && (2) \\
&= \sum_a \pi(a \mid s) \sum_{s'} \sum_r p(s', r \mid s, a) \Big[ r + \gamma\, \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] \Big] && (3) \\
&= \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma\, v_\pi(s') \big], \quad \text{for all } s \in \mathcal{S} && (4)
\end{aligned}
$$

Let's go through it step by step.

**(1) → (2).** The return has a recursive structure. Pull the first reward out of the sum, and what is left is just the return from the next step, discounted once:

$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \\
&= R_{t+1} + \gamma\big(R_{t+2} + \gamma R_{t+3} + \cdots\big) \\
&= R_{t+1} + \gamma G_{t+1}
\end{aligned}
$$


So "everything from now on" = "the next reward" + $\gamma$ × "everything from the next step on".

**(2) → (3).** The expectation is an average over everything random that happens in one step. Starting in state $s$, two random things happen:

1. Maurice picks an action $a$, with probability $\pi(a \mid s)$ (his policy).
2. The maze responds with a next state $s'$ and a reward $r$, with probability $p(s', r \mid s, a)$.

So we average over both: we weight every possible $(a, s', r)$ combination by its probability $\pi(a \mid s)\, p(s', r \mid s, a)$, and for each one, what we get is the reward $r$ plus $\gamma$ times the expected return from wherever we landed, $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$. (Why can we condition only on $s'$ and forget $s$ and $a$? The Markov property again: once you know where Maurice is now, how he got there does not change what happens next.)

Going from (2) to (3) might have felt like a big jump, let's make it simple. We need just one tool, the **law of total expectation**: to find an average, you can split the world into cases, find the average within each case, and then take a weighted mean of those averages using how likely each case is:

$$
\mathbb{E}[X \mid Y] = \sum_z P(Z = z \mid Y)\, \mathbb{E}[X \mid Y, Z = z]
$$

(Example: the average height in a class = (fraction of girls × average height of girls) + (fraction of boys × average height of boys).)

> Again, let's go through it. Think of it as "an average of averages, weighted by how likely each case is".
>
> Let's go back to the casino for a second. Say every evening you play machine A with probability $0.7$ and machine B with probability $0.3$. Machine A pays out $2$ on average and machine B pays out $10$ on average. What do you make on an average evening?
>
> $$\mathbb{E}[\text{payout}] = \underbrace{0.7}_{P(\text{A})} \times \underbrace{2}_{\mathbb{E}[\text{payout} \mid \text{A}]} + \underbrace{0.3}_{P(\text{B})} \times \underbrace{10}_{\mathbb{E}[\text{payout} \mid \text{B}]} = 1.4 + 3 = 4.4$$
>
> Notice what we did *not* need: the full list of every possible payout and its probability. We only needed the average *within* each case, and how likely each case is. That is the whole trick.
>
> **Why is it true?** Starting from the definition of expected value from the beginning, $\mathbb{E}[X] = \sum_x x \, P(X = x)$:
>
> $$
> \begin{aligned}
> \mathbb{E}[X] &= \sum_x x \, P(X = x) && \text{(definition of expected value)} \\
> &= \sum_x x \sum_z P(X = x, Z = z) && \text{(split } P(X = x) \text{ over every case } z\text{)} \\
> &= \sum_x x \sum_z P(Z = z)\, P(X = x \mid Z = z) && \text{(conditional probability, rearranged)} \\
> &= \sum_z P(Z = z) \sum_x x \, P(X = x \mid Z = z) && \text{(swap the order of the sums)} \\
> &= \sum_z P(Z = z)\, \mathbb{E}[X \mid Z = z] && \text{(the inner sum is the average within case } z\text{)}
> \end{aligned}
> $$
>
> The third line is just the Venn diagram formula $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$ multiplied out: $P(A \cap B) = P(B)\, P(A \mid B)$.
>
> If we already know something, say $Y$ (for us, $S_t = s$), nothing changes. Every probability and expectation just gets a "$\mid Y$" attached, which gives the version written above.
>
> **One catch:** the cases $z$ must cover every possibility, and no two of them can happen at the same time (machine A *or* machine B each evening, never both and never neither). Otherwise the weights don't add up to 1 and the average comes out wrong.
>
> **In our maze**, the "something we already know" is $S_t = s$, and we split twice:
> - first on Maurice's action: the cases are the actions $a$, weighted by $\pi(a \mid s)$,
> - then on the maze's response: the cases are the $(s', r)$ pairs, weighted by $p(s', r \mid s, a)$.

We apply it twice, first splitting on the action, then splitting on what the maze does:

$$
\begin{aligned}
&\mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s] && (2) \\
&= \sum_a \pi(a \mid s)\; \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] && (2a) \\
&= \sum_a \pi(a \mid s) \sum_{s'} \sum_r p(s', r \mid s, a)\; \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s', R_{t+1} = r] && (2b) \\
&= \sum_a \pi(a \mid s) \sum_{s'} \sum_r p(s', r \mid s, a) \Big[ r + \gamma\, \mathbb{E}_\pi[G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s', R_{t+1} = r] \Big] && (2c) \\
&= \sum_a \pi(a \mid s) \sum_{s'} \sum_r p(s', r \mid s, a) \Big[ r + \gamma\, \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] \Big] && (3)
\end{aligned}
$$

- **(2) → (2a):** split on which action Maurice picks. The chance of each case is $\pi(a \mid s)$.
- **(2a) → (2b):** within each action, split again on where the maze sends him and what reward it gives. The chance of each case is $p(s', r \mid s, a)$.
- **(2b) → (2c):** inside the expectation we now *know* $R_{t+1} = r$, so it is no longer random and comes out as just $r$ (the average of a known number is the number itself). The $\gamma$ also comes out, since the expectation of a constant times something is the constant times its expectation.
- **(2c) → (3):** the Markov property. The future return $G_{t+1}$ only depends on where Maurice is at $t+1$, so knowing $s$, $a$ and $r$ on top of $s'$ tells us nothing new, and we can drop them.

**(3) → (4).** Look at $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$. It is "the expected return when starting from state $s'$ and following $\pi$", which is exactly the definition of $v_\pi(s')$! So we swap it in. (We also write $\sum_{s'}\sum_r$ as $\sum_{s',r}$ to save some ink.)

And that is the magic: the value of a state is now written in terms of the immediate reward plus the discounted values of the states right after it. We no longer need to sum over the infinite future, we just look one step ahead.

This is popularly called the **Bellman equation** for $v_\pi$ (the state-value function).

We have the map and we have the Bellman equation, but we still need an algorithm to actually compute these values, right? How do we do that?

This family of methods, where you use a perfect model of the world (our map, i.e. $p$) to compute values by repeatedly applying the Bellman equation, is called **Dynamic Programming (DP)**. Notice that Maurice never actually walks the maze here, all of it is done by *thinking* with the map. This is also called **planning**.

The first piece is what we call **policy evaluation**: given a policy $\pi$, compute $v_\pi$. The trick is to turn the Bellman equation into an update rule. Start with arbitrary guesses for $V(s)$, then go through all the states, replacing each $V(s)$ with the right-hand side of the Bellman equation computed using the current guesses. Keep going until the values stop changing.

```
Iterative Policy Evaluation, for estimating V ≈ v_π

Input: π, the policy to be evaluated
Algorithm parameter: a small threshold θ > 0 determining accuracy of estimation
Initialize V(s) arbitrarily, for all s ∈ S, except that V(terminal) = 0

Loop:
    Δ ← 0
    Loop for each s ∈ S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
        Δ ← max(Δ, |v − V(s)|)
until Δ < θ
```

Taken from Sutton and barto [ACTUALLY_ADD_THE_IMAGE]

> In code, it looks something like this. To keep things small, we use the classic 4×4 grid from the book: the exits are the top-left and bottom-right corners, every step costs $-1$, and Maurice follows the random policy (each direction with probability $0.25$).

```python
import numpy as np

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
TERMINALS = {0, 15}  # the two exits: top-left and bottom-right corners
ACTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
GAMMA = 1.0          # no discounting needed, every step already costs -1

def next_state(state, action):
    row, col = divmod(state, GRID_SIZE)
    d_row, d_col = ACTIONS[action]
    # walking into a wall leaves you where you are
    row = min(max(row + d_row, 0), GRID_SIZE - 1)
    col = min(max(col + d_col, 0), GRID_SIZE - 1)
    return row * GRID_SIZE + col

def reward(state, action):
    return -1  # every step hurts, so the shortest way out wins

def policy_evaluation(policy, theta=1e-4):
    V = np.zeros(N_STATES)  # V(terminal) stays 0 forever
    while True:
        delta = 0
        for s in range(N_STATES):
            if s in TERMINALS:
                continue
            v = V[s]
            # the map is deterministic, so Σ_{s',r} p(s',r|s,a) collapses to a single next state
            V[s] = sum(prob * (reward(s, a) + GAMMA * V[next_state(s, a)])
                       for a, prob in policy[s].items())
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            return V

# the equiprobable random policy: every action with probability 0.25
random_policy = {s: {a: 0.25 for a in ACTIONS} for s in range(N_STATES)}

V = policy_evaluation(random_policy)
print(V.reshape(GRID_SIZE, GRID_SIZE).round(1))
```

```
[[  0. -14. -20. -22.]
 [-14. -18. -20. -20.]
 [-20. -20. -18. -14.]
 [-22. -20. -14.   0.]]
```

Each number is "how many steps, on average, will a Maurice who wanders around *randomly* need to get out from here" (negated, since every step costs $-1$). Squares near an exit are worth more, squares far from both exits are worth the least.

This gives us the value of every state under the policy, but now we need to run Maurice on it, so he can find the values and follow them. "Following them" means that in every state, Maurice picks the action that leads to the best $r + \gamma V(s')$ (this is called **policy improvement**). But once the policy changes, its values change too, so we evaluate again, improve again, and keep going until the policy stops changing. This is called **policy iteration**:


```
Policy Iteration (using iterative policy evaluation) for estimating π ≈ π*

1. Initialization
   V(s) ∈ ℝ and π(s) ∈ A(s) arbitrarily for all s ∈ S; V(terminal) = 0

2. Policy Evaluation
   Loop:
       Δ ← 0
       Loop for each s ∈ S:
           v ← V(s)
           V(s) ← Σ_{s',r} p(s',r|s,π(s)) [r + γ V(s')]
           Δ ← max(Δ, |v − V(s)|)
   until Δ < θ (a small positive number determining the accuracy of estimation)

3. Policy Improvement
   policy-stable ← true
   For each s ∈ S:
       old-action ← π(s)
       π(s) ← argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
       If old-action ≠ π(s), then policy-stable ← false
   If policy-stable, then stop and return V ≈ v* and π ≈ π*; else go to 2
```

> Building on the code above, policy iteration is just a loop around `policy_evaluation`:

```python
def greedy_action(V, s):
    # the action with the best one-step lookahead: r + γ V(s')
    return max(ACTIONS, key=lambda a: reward(s, a) + GAMMA * V[next_state(s, a)])

def policy_iteration():
    policy = random_policy  # 1. Initialization: start with the random policy
    while True:
        V = policy_evaluation(policy)  # 2. Policy Evaluation
        # 3. Policy Improvement: in every state, put all the probability on the greedy action
        new_policy = {s: {greedy_action(V, s): 1.0} for s in range(N_STATES)}
        if new_policy == policy:  # policy-stable, we are done
            return V, policy
        policy = new_policy

ARROWS = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

def show(V, policy):
    print(V.reshape(GRID_SIZE, GRID_SIZE).round(1))
    for row in range(GRID_SIZE):
        print(' '.join('■' if s in TERMINALS else ARROWS[next(iter(policy[s]))]
                       for s in range(row * GRID_SIZE, (row + 1) * GRID_SIZE)))

V, policy = policy_iteration()
show(V, policy)
```

```
[[ 0. -1. -2. -3.]
 [-1. -2. -3. -2.]
 [-2. -3. -2. -1.]
 [-3. -2. -1.  0.]]
■ ← ← ↓
↑ ↑ ↑ ↓
↑ ↑ ↓ ↓
↑ → → ■
```

Now every value is exactly minus the number of steps to the nearest exit, and the arrows show Maurice the shortest way out from every square.

We let Maurice go wild after telling him that he has to follow this algorithm.

But the problem that we realise is, Maurice is taking far too long! Because every round of policy evaluation sweeps through the whole maze again and again until the values have fully settled, and only then do we improve the policy a little. It would be much better if, in every sweep, Maurice directly used the value of the best action (the max) instead of waiting for the values of the current policy to settle. That way evaluation and improvement happen together in a single sweep. This is called **value iteration** and we can implement it as such!

```
Value Iteration, for estimating π ≈ π*

Algorithm parameter: a small threshold θ > 0 determining accuracy of estimation
Initialize V(s), for all s ∈ S⁺, arbitrarily except that V(terminal) = 0

Loop:
    Δ ← 0
    Loop for each s ∈ S:
        v ← V(s)
        V(s) ← max_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
        Δ ← max(Δ, |v − V(s)|)
until Δ < θ

Output a deterministic policy, π ≈ π*, such that
    π(s) = argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
```

The above algo can be implemented as

```python
def value_iteration(theta=1e-4):
    V = np.zeros(N_STATES)
    while True:
        delta = 0
        for s in range(N_STATES):
            if s in TERMINALS:
                continue
            v = V[s]
            # the ONLY change from policy evaluation: max over actions instead of a π-weighted sum
            V[s] = max(reward(s, a) + GAMMA * V[next_state(s, a)] for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # output a deterministic policy: act greedily with respect to the final values
    policy = {s: {greedy_action(V, s): 1.0} for s in range(N_STATES)}
    return V, policy

V, policy = value_iteration()
show(V, policy)
```

```
[[ 0. -1. -2. -3.]
 [-1. -2. -3. -2.]
 [-2. -3. -2. -1.]
 [-3. -2. -1.  0.]]
■ ← ← ↓
↑ ↑ ↑ ↓
↑ ↑ ↓ ↓
↑ → → ■
```

Put `policy_evaluation` and `value_iteration` side by side and you will notice they are almost the same function. The only difference is one line:

- **Policy evaluation:** a square's new value is the *average* over actions, weighted by how likely Maurice is to take each one ($\sum_a \pi(a \mid s) \ldots$).
- **Value iteration:** a square's new value is the value of the *best* action ($\max_a \ldots$).

That is the whole idea. Instead of asking "how good is this square if Maurice keeps doing what he is doing?", value iteration asks "how good is this square if Maurice does the best thing from here?". Each sweep, good news (being close to an exit) spreads one square further out, like ripples in a pond, until every square knows its shortest distance out. Then Maurice just follows the arrows. And we get the same answer as policy iteration, without ever having to fully evaluate a policy.

You stick this new algo inside of Maurice, he performs superbly and gets you the best path in only 5 iterations. Now you follow it, dancing and frog-leaping in happiness, because you have made so much money and have ESCAPED!!! with your freedom. As you reach near the gates of Saintsbury, you see a sight. A sight that shakes you, that mortifies you with fear!!

"Oh no, it is the manager!!!"

"No you fool, I am the manager's brother. John!"

"Papa John???"

"What, no! Stop this malarkey. Anyhoo, if you wish to exit, you must answer this query of mine..."

He pulls out a crumpled sheet of paper. It is the grid of values your Maurice computed earlier, back when he was wandering around *randomly* (squares numbered 0 to 15, left to right, top to bottom, with the exits at 0 and 15):

| | | | |
|:---:|:---:|:---:|:---:|
| **0** <br> exit | **1** <br> $-14$ | **2** <br> $-20$ | **3** <br> $-22$ |
| **4** <br> $-14$ | **5** <br> $-18$ | **6** <br> $-20$ | **7** <br> $-20$ |
| **8** <br> $-20$ | **9** <br> $-20$ | **10** <br> $-18$ | **11** <br> $-14$ |
| **12** <br> $-22$ | **13** <br> $-20$ | **14** <br> $-14$ | **15** <br> exit |

"Say Maurice is standing on square 11. Instead of wandering, he makes *one* deliberate move, down, and only *then* goes back to wandering randomly like a fool. What is that move worth? And what if he is on square 7 and moves down?"

Wow, that is some question, quite perplexing if I say so myself. But our hero is left undaunted. You got this, let's think it through, what do we know?

First, what do the numbers on the sheet mean? Each one is $v_\pi(s)$: what a square is worth if Maurice wanders randomly from there on. But John is asking something slightly different. He is fixing the *first* move, and only after that does Maurice go back to following $\pi$.

So let's do exactly what the Bellman equation taught us: look one step ahead. Making a move earns the immediate reward, plus the value of wherever we land. The map is deterministic and $\gamma = 1$, so:

$$
\text{value of the move} = -1 + v_\pi(\text{square we land on})
$$

- **Square 11, down:** we land on square 15, the exit, worth $0$. So the move is worth $-1 + 0 = -1$.
- **Square 7, down:** we land on square 11, worth $-14$. So the move is worth $-1 + (-14) = -15$.

"Correct!" says John, visibly annoyed.

And without realising it, you have just discovered a new quantity. The value of *taking action $a$ in state $s$, and following $\pi$ afterwards* is called the **action-value function**, written $q_\pi(s, a)$:

$$
q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\,\big[r + \gamma\, v_\pi(s')\big]
$$

> Want to double-check your answer? Square 11's value should be the *average* of the values of its four moves, since random Maurice picks each one with probability $0.25$. From square 11: down → exit gives $-1$, up → square 7 gives $-1 + (-20) = -21$, left → square 10 gives $-1 + (-18) = -19$, and right bumps into the wall and stays on 11, giving $-1 + (-14) = -15$. The average is $\frac{-1 - 21 - 19 - 15}{4} = \frac{-56}{4} = -14$, exactly $v_\pi(11)$! In general, $v_\pi(s) = \sum_a \pi(a \mid s)\, q_\pi(s, a)$: the value of a state is the average value of the moves you might make from it.

Our hero triumphs once again! You have been through numerous challenges, and you walk out of the gates of Saintsbury only to find out... it was all a ruse!!! No wonder the map was so conveniently placed there.

The manager is a mischievous man, he is playing with you. He is enjoying putting you through all this misery. But fret not, these little encumbrances will not shake your willpower.

This time we have no map, and the maze is as complex as it can be....


AND that's all folks, join in for the next article to find out how our hero escapes this problem.

## A few loose ends

Our hero moved fast, so there are a few ideas we rushed past. They are worth a minute each, because everything from the next article onwards builds on them.

### The *optimal* way of acting

Remember when we said "we will talk about the optimal way of acting in a bit"? Here it is. Out of all the possible policies, the best one, $\pi_\ast$, is the one whose values are the highest in every state. We call those values the **optimal value functions**:

$$
v_*(s) = \max_\pi v_\pi(s), \qquad q_*(s, a) = \max_\pi q_\pi(s, a)
$$

If you already knew $q_\ast$, acting optimally would be trivial: in every state, pick the move worth the most, $v_\ast(s) = \max_a q_\ast(s, a)$. Plugging that into the Bellman equation gives the **Bellman optimality equation**:

$$
v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\,\big[r + \gamma\, v_*(s')\big]
$$

Compare it to the Bellman equation for $v_\pi$. The only thing that changed is that the $\pi$-weighted average, $\sum_a \pi(a \mid s)$, became a $\max_a$. Look familiar? That is *exactly* the one line that turned `policy_evaluation` into `value_iteration`. Value iteration is just the Bellman optimality equation turned into an update rule.

### Why does policy improvement always help?

In policy iteration we kept making Maurice greedy and trusted that this never makes things worse. John's question shows why. Random Maurice on square 11 is worth $v_\pi(11) = -14$, but moving down once and *then* wandering is worth $q_\pi(11, \text{down}) = -1$. If making the better move *once* helps, then making it *every time* you are on that square can only help more. This is the **policy improvement theorem**: if $q_\pi(s, \pi'(s)) \ge v_\pi(s)$ in every state, then the new policy $\pi'$ is at least as good as $\pi$ everywhere. And since the greedy action is the *best* of the moves, it is always at least as good as their average, $v_\pi(s)$. Because there is only a finite number of policies and each round never makes things worse, policy iteration has to stop, and when it does, the policy is optimal.

### The big picture: generalized policy iteration

Step back and look at what we have been doing. There are always two processes pulling on each other:

- **Evaluation:** make the values match the current policy.
- **Improvement:** make the policy greedy with respect to the current values.

Each one changes the ground under the other: a new policy makes the old values wrong, and new values make the old policy not greedy anymore. But they settle down together at exactly one place, the optimal policy and its values. Policy iteration runs evaluation all the way to the end before improving. Value iteration does just one sweep of evaluation before improving. Anything in between works too. This idea is called **generalized policy iteration (GPI)**, and almost every RL algorithm you will ever meet, including the ones in the next article where we lose the map, is some version of it.

## Where to go from here

If you would like, I will recommend reading [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Sutton and Barto (it is free online!). You should have all the background needed to make sense of it now. If you do run into some issues and have trouble understanding, TELL ME, that will help me understand what exactly it was that I could not encapsulate.

Now, if I have helped you, even as a mere spectator, through an arduous journey filled with perils, laughs, cries and joy, then I have but one request: consider sharing this with your friends, so they can go on a very cool and fun journey as well!



-----

<!-- 

These are my personal notes from MSAI635 (Reinforcement Learning) here at UMD, as I go through the book *Reinforcement Learning: An Introduction* by Richard Sutton and Andrew Barto.

I will try to explain each chapter as I understood it, as well as do the exercises!


## Chapter 2: k-armed Bandits

Exercise 2.1 In $\varepsilon$-greedy action selection, for the case of two actions and $\varepsilon = 0.5$, what is
the probability that the greedy action is selected?

Ans -> I believe it will be:

$$P(\text{greedy}) = (1-\varepsilon) \cdot 1 + \varepsilon \cdot \frac{1}{n}$$

where $n$ is the number of actions. With probability $(1-\varepsilon)$ we exploit and pick the greedy action for sure (probability 1). With probability $\varepsilon$ we explore, picking uniformly at random among all $n$ actions (including the greedy one), so the greedy action has a $1/n$ chance there too.

Plugging in $\varepsilon = 0.5$, $n = 2$:

$$P(\text{greedy}) = 0.5 \cdot 1 + 0.5 \cdot \frac{1}{2} = 0.5 + 0.25 = 0.75$$

Exercise 2.2: Bandit example Consider a $k$-armed bandit problem with $k = 4$ actions,
denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using
$\varepsilon$-greedy action selection, sample-average action-value estimates, and initial estimates
of $Q_1(a) = 0$, for all $a$. Suppose the initial sequence of actions and rewards is $A_1 = 1$,
$R_1 = -1$, $A_2 = 2$, $R_2 = 1$, $A_3 = 2$, $R_3 = -2$, $A_4 = 2$, $R_4 = 2$, $A_5 = 3$, $R_5 = 0$. On some
of these time steps the $\varepsilon$ case may have occurred, causing an action to be selected at
random. On which time steps did this definitely occur? On which time steps could this
possibly have occurred?

Ans -> I believe A2 and A5 are when the random epsilon occurred, as at these times the greedy action was not taken!

**OPUS 5.5 correction ->** The PDF copy dropped two minus signs: it is $R_1 = -1$ and $R_3 = -2$. Redoing it with the correct rewards (sample averages, starting from $Q = [0, 0, 0, 0]$):

| Step | $Q$ before acting | Greedy action(s) | Taken | Verdict |
|---|---|---|---|---|
| 1 | $[0, 0, 0, 0]$ | 1, 2, 3, 4 (tie) | 1 | possibly random |
| 2 | $[-1, 0, 0, 0]$ | 2, 3, 4 (tie) | 2 | possibly random |
| 3 | $[-1, 1, 0, 0]$ | 2 | 2 | possibly random |
| 4 | $[-1, -0.5, 0, 0]$ | 3, 4 (tie) | 2 | **definitely random** |
| 5 | $[-1, 1/3, 0, 0]$ | 2 | 3 | **definitely random** |

So the $\varepsilon$ case *definitely* occurred at steps 4 and 5, and *could possibly* have occurred at every step (1 through 5). A random pick can land on the greedy action by chance, so any step that looks greedy might still have been an exploration step.

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

Exercise 2.4 If the step-size parameters, $\alpha_n$, are not constant, then the estimate $Q_n$ is
a weighted average of previously received rewards with a weighting different from that
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
difficulties that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the $q_\ast(a)$ start out equal and then take
independent random walks (say by adding a normally distributed increment with mean 0
and standard deviation 0.01 to all the $q_\ast(a)$ on each step). Prepare plots like Figure 2.2
for an action-value method using sample averages, incrementally computed, and another
action-value method using a constant step-size parameter, $\alpha = 0.1$. Use $\varepsilon = 0.1$ and
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
        reward = rng.normal(loc=Q[action], scale=1)
        self.q[action] += (1 / self.n[action]) * (reward - self.q[action])
        return reward

    def constant_step_size(self, action, alpha=0.1):
        self.n[action] += 1
        reward = rng.normal(loc=Q[action], scale=1)
        self.q[action] += alpha * (reward - self.q[action])
        return reward

reward_sum_1 = np.zeros(10000)
reward_sum_2 = np.zeros(10000)
optimal_count_1 = np.zeros(10000)
optimal_count_2 = np.zeros(10000)

for j in range(2000):
    agent_1 = Agent()
    agent_2 = Agent()

    Q = [5.0] * 10
    Q = np.asarray(Q)

    for i in range(10000):
        Q = step(Q)
        value_1 = random.random()
        value_2 = random.random()

        if value_1 > epsilon:
            action_1 = agent_1.q.index(max(agent_1.q))
        else:
            action_1 = random.randint(0, 9)

        if value_2 > epsilon:
            action_2 = agent_2.q.index(max(agent_2.q))
        else:
            action_2 = random.randint(0, 9)

        reward_1 = agent_1.sample_average(action_1)
        reward_2 = agent_2.constant_step_size(action_2)

        reward_sum_1[i] += reward_1
        reward_sum_2[i] += reward_2

        if action_1 == Q.argmax():
            optimal_count_1[i] += 1

        if action_2 == Q.argmax():
            optimal_count_2[i] += 1

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

plt.plot(pct_optimal_1, label="sample average")
plt.plot(pct_optimal_2, label="constant step-size")
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.legend()
plt.show()
```


Exercise 2.6: Mysterious Spikes The results shown in Figure 2.3 should be quite reliable
because they are averages over 2000 individual, randomly chosen 10-armed bandit tasks.
Why, then, are there oscillations and spikes in the early part of the curve for the optimistic
method? In other words, what might make this method perform particularly better or
worse, on average, on particular early steps?

Ans -> As the number of bandits is restricted to 10, the optimistic one is going to try all 10 of them, and one of them is likely to be the most optimal action, and equally one is likely to be the least optimal action. That is why we see oscillations early on.

**OPUS 5.5 correction ->** Partly right, but the key mechanism is what happens *after* the first round. With $Q_1(a) = +5$ and purely greedy selection, the first ~10 steps try every arm once (each pull drags that arm's estimate below $5$, so an untried arm always wins). After that round, the arm whose estimate dropped the *least* is most likely the truly best arm, so greedy picks it, and that produces the spike (around step 11). But one pull doesn't bring the estimate down to its true value (with $\alpha = 0.1$ it moves only 10% of the way), so pulling it again pushes it below the other arms' still-inflated estimates. Greedy then switches away to the other arms, which causes the dip. The "every arm is still optimistic" effect wears off over the next few rounds, so the oscillation dies down. It shows up in the 2000-run *average* because every run goes through the same "try all, then pick the best" schedule at the same steps.


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
identifying for each its states, actions, and rewards. Make the three examples as different
from each other as possible. The framework is abstract and flexible and can be applied in
many different ways. Stretch its limits in some way in at least one of your examples.

Ans.
1. A pizza making robot: the state can be the current state of the pizza, the actions can be the amount of each ingredient to put on, and the reward will be whether the user liked it or not.

2. A water heater: the action is heating up the copper wire, the state is the current temperature of the water, and the reward is how close it is to the expected water temperature.

3. Teaching a one-legged robot to walk: the motor commands are the actions, the current position and whether it is upright is the state, and the distance traveled is the reward.


Exercise 3.2 Is the MDP framework adequate to usefully represent all goal-directed
learning tasks? Can you think of any clear exceptions?

Ans. No, it is not, as it expects that the current state depends exclusively on the previous state and disregards all the history prior to that. This can fail in tasks where all the past states are important. An example can be an equity bot which sells equity: the worth of an equity cannot be directly valued by what it was in the previous state, we also have to look at when it was bought and for how much, depending on a prior step!

**OPUS 5.5 correction ->** The Markov property is a property of *how you define the state*, not of the problem itself. In the equity example you can include the purchase price and time in the state, and then it is Markov again. So that is not a true exception. Better exceptions are:
- **Partial observability**: the agent cannot observe the information it would need (e.g. poker, where opponents' cards are hidden), so no state it can build is Markov.
- **Goals that cannot be reduced to a single scalar reward**: multiple conflicting objectives, or goals defined over whole trajectories (e.g. "visit A *before* B") that a per-step scalar reward cannot express cleanly.


Exercise 3.3 Consider the problem of driving. You could define the actions in terms of
the accelerator, steering wheel, and brake, that is, where your body meets the machine.
Or you could define them farther out—say, where the rubber meets the road, considering
your actions to be tire torques. Or you could define them farther in—say, where your
brain meets your body, the actions being muscle twitches to control your limbs. Or you
could go to a really high level and say that your actions are your choices of where to drive.
What is the right level, the right place to draw the line between agent and environment?
On what basis is one location of the line to be preferred over another? Is there any
fundamental reason for preferring one location over another, or is it a free choice? 

Ans. Let us first look at what does not work.

Rubber meets the road obviously does not work, as we have no control over that and we cannot have well-defined actions over it!

Brain meets body, well, it does provide actions, but they cannot be constrained. It has far too many variables.

Where to drive is not the right level either, because the destination is constant while the paths to reach it are multiple. We have to define a level at which we have control, defined variables, and something that we can optimize.

After this process of elimination, the only reasonable answer left is actions in terms of brakes, acceleration...

**OPUS 5.5 correction ->** The book's point is that there is no single right level. It is largely a free choice that depends on the task. The general rule is that the agent–environment boundary is drawn at the **limit of the agent's absolute control**: anything the agent cannot change arbitrarily belongs to the environment. All four levels can be valid:
- Tire torques are directly controllable for a self-driving car's low-level controller.
- Muscle twitches are the right level if you are modelling a human motor-control system.
- "Where to drive" is the right level for a route-planning agent, which hands the actual driving off to a lower-level controller.

What makes one level better than another is practical: the actions should be ones the agent can actually execute reliably, and the level should match the decisions you want to learn.


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
discounting, with all rewards zero except for $-1$ upon failure. What then would the
return be at each time? How does this return differ from that in the discounted, continuing
formulation of this task?

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
no improvement in escaping from the maze. What is going wrong? Have you effectively
communicated to the agent what you want it to achieve?

[Ans.] No, because the agent has no metric of knowing if it has improved over time. Nor any other incentive to do so.

**OPUS 5.5 correction ->** The more precise reason: the goal is to maximize the *undiscounted* total reward (3.7), and the only reward is $+1$ at the exit. So **every policy that eventually escapes gets exactly the same return, $G_0 = 1$**, whether it takes 10 steps or 10,000. From the agent's point of view, a random wander that eventually stumbles out is already optimal, so there is nothing to improve. You have told it *that* escaping is good, but not that escaping *quickly* is better. Fixes:
- Give $-1$ per time step, so a shorter path means a higher return, or
- Use discounting ($\gamma < 1$), so a $+1$ received later is worth less: $G_0 = \gamma^{T-1}$.

Exercise 3.8 Suppose $\gamma = 0.5$ and the following sequence of rewards is received $R_1 = -1$,
$R_2 = 2$, $R_3 = 6$, $R_4 = 3$, and $R_5 = 2$, with $T = 5$. What are $G_0, G_1, \dots, G_5$? Hint:
Work backwards.

[Ans.] Working backwards with $G_t = R_{t+1} + \gamma G_{t+1}$:

- $G_5 = 0$
- $G_4 = R_5 + \gamma G_5 = 2 + 0.5(0) = 2$
- $G_3 = R_4 + \gamma G_4 = 3 + 0.5(2) = 4$
- $G_2 = R_3 + \gamma G_3 = 6 + 0.5(4) = 8$
- $G_1 = R_2 + \gamma G_2 = 2 + 0.5(8) = 6$
- $G_0 = R_1 + \gamma G_1 = -1 + 0.5(6) = 2$


Exercise 3.9 Suppose $\gamma = 0.9$ and the reward sequence is $R_1 = 2$ followed by an infinite
sequence of 7s. What are $G_1$ and $G_0$?

[Ans.] $G_1 = R_2 + \gamma R_3 + \gamma^2 R_4 + \dots = 7(1+\gamma+\gamma^2+\dots) = \dfrac{7}{1-\gamma} = \dfrac{7}{0.1} = 70$

$G_0 = R_1 + \gamma G_1 = 2 + 0.9(70) = 2 + 63 = 65$


Exercise 3.10 Prove the second equality in (3.10).

Ans -> Sum of GP

**[IMPORTANT]** Exercise 3.11 If the current state is $S_t$, and actions are selected according to a stochastic
policy $\pi$, then what is the expectation of $R_{t+1}$ in terms of $\pi$ and the four-argument
function $p$ (3.2)?

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
$v_\pi$ shown in Figure 3.2 (right) of Example 3.5. Show numerically that this equation holds
for the center state, valued at $+0.7$, with respect to its four neighboring states, valued at
$+2.3$, $+0.4$, $-0.4$, and $+0.7$. (These numbers are accurate only to one decimal place.)

[Ans.] Under the equiprobable random policy $\pi(a\mid s)=0.25$ for each of the four actions; each action deterministically moves to one neighboring cell with reward $r=0$; $\gamma=0.9$:

$$v_\pi(s) = \sum_a \pi(a\mid s)\big[r+\gamma v_\pi(s')\big] = 0.25 \times 0.9 \times (2.3+0.4-0.4+0.7)$$

$$= 0.225 \times 3.0 = 0.675 \approx 0.7$$

which matches the stated value of $+0.7$ (to the one-decimal-place accuracy given).

Exercise 3.15 In the gridworld example, rewards are positive for goals, negative for
running into the edge of the world, and zero the rest of the time. Are the signs of these
rewards important, or only the intervals between them? Prove, using (3.8), that adding a
constant $c$ to all the rewards adds a constant, $v_c$, to the values of all states, and thus
does not affect the relative values of any states under any policies. What is $v_c$ in terms
of $c$ and $\gamma$?

[Ans.] Only the intervals between rewards matter, not their absolute signs — adding a constant $c$ shifts every state's value by the same fixed amount, so the relative ordering of states (and hence which policy is better than which) is unchanged.

Proof, using (3.8): replacing every reward $R_{t+k+1}$ with $R_{t+k+1}+c$ gives a new return

$$G_t' = \sum_{k=0}^{\infty}\gamma^k(R_{t+k+1}+c) = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1} + \sum_{k=0}^{\infty}\gamma^k c = G_t + c\sum_{k=0}^{\infty}\gamma^k$$

The second sum is a geometric series (as in Exercise 3.9/3.10), so for $\gamma<1$:

$$v_c = c\sum_{k=0}^{\infty}\gamma^k = \frac{c}{1-\gamma}$$

So $G_t' = G_t + v_c$ for every state and every policy — the shift is identical everywhere, so it adds the same constant to $v_\pi(s)$ for every $s$ under every $\pi$, leaving all relative comparisons between states/policies unaffected.

Exercise 3.16 Now consider adding a constant c to all the rewards in an episodic task,
such as maze running. Would this have any effect, or would it leave the task unchanged
as in the continuing task above? Why or why not? Give an example.

[Ans.] Unlike the continuing case, this does have an effect — the sign of $c$ matters here.

In the continuing case, the constant contribution to the return was $v_c = c\sum_{k=0}^{\infty}\gamma^k = c/(1-\gamma)$, the *same* value regardless of policy, since every policy's return sums to infinity. In the episodic case the sum instead stops at the episode's termination time $T$:

$$G_0' = G_0 + c\sum_{k=0}^{T-1}\gamma^k$$

and this constant contribution now depends on $T$ — the episode length — which varies from one policy/trajectory to another. So the shift is no longer uniform across policies, and it can change which policy looks best.

Concretely, in the maze example, rewards are $0$ per step and $+1$ upon escaping. Adding $c>0$ to every reward turns this into $c$ per step and $1+c$ upon escaping — so every extra time step spent wandering before escaping now earns an additional $+c$. This gives the agent an incentive to *prolong* the episode rather than escape quickly (or, in the undiscounted case, to never escape at all, accumulating $c$ indefinitely), which directly undermines the original goal of finding the exit as fast as possible.

**[IMPORTANT]** Exercise 3.17 What is the Bellman equation for action values, that
is, for $q_\pi$? It must give the action value $q_\pi(s, a)$ in terms of the action
values, $q_\pi(s', a')$, of possible successors to the state–action pair $(s, a)$.
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

*[Backup diagram: root state $s$ with value $v_\pi(s)$, branching to actions $a_1, a_2, a_3$, each taken with probability $\pi(a \mid s)$, with leaf values $q_\pi(s, a)$.]*

Give the equation corresponding to this intuition and diagram for the value at the root
node, $v_\pi(s)$, in terms of the value at the expected leaf node, $q_\pi(s, a)$, given $S_t = s$. This
equation should include an expectation conditioned on following the policy, $\pi$. Then give
a second equation in which the expected value is written out explicitly in terms of $\pi(a \mid s)$
such that no expected value notation appears in the equation.

[Ans.] The root's value is the average of the leaf values $q_\pi(s,a)$, where the averaging is over which action $A_t$ the policy randomly picks:

$$v_\pi(s) = \mathbb{E}_\pi\big[q_\pi(s, A_t) \mid S_t=s\big]$$

Writing that expectation out explicitly, weighting each leaf $q_\pi(s,a)$ by the probability $\pi(a\mid s)$ of the policy choosing that branch:

$$v_\pi(s) = \sum_a \pi(a\mid s)\, q_\pi(s,a)$$


**[IMPORTANT]** Exercise 3.19 The value of an action, $q_\pi(s, a)$, depends on the expected next reward and
the expected sum of the remaining rewards. Again we can think of this in terms of a
small backup diagram, this one rooted at an action (state–action pair) and branching to
the possible next states:

*[Backup diagram: root state–action pair $(s, a)$ with value $q_\pi(s, a)$, branching via expected rewards $r_1, r_2, r_3$ to next states $s'_1, s'_2, s'_3$ with values $v_\pi(s')$.]*

Give the equation corresponding to this intuition and diagram for the action value,
$q_\pi(s, a)$, in terms of the expected next reward, $R_{t+1}$, and the expected next state value,
$v_\pi(S_{t+1})$, given that $S_t = s$ and $A_t = a$. This equation should include an expectation but
not one conditioned on following the policy. Then give a second equation, writing out the
expected value explicitly in terms of $p(s', r \mid s, a)$ defined by (3.2), such that no expected
value notation appears in the equation.

[Ans.] With $S_t=s$ and $A_t=a$ both already fixed, the only remaining randomness is the environment's response — no $\pi$ needed:

$$q_\pi(s,a) = \mathbb{E}\big[R_{t+1} + \gamma\, v_\pi(S_{t+1}) \mid S_t=s, A_t=a\big]$$

($v_\pi(S_{t+1})$ replaces $G_{t+1}$ here by the law of iterated expectations: $\mathbb{E}[G_{t+1}\mid S_{t+1}=s']=v_\pi(s')$ from Exercise 3.13, so averaging $v_\pi(S_{t+1})$ over $S_{t+1}$ gives the same result as averaging $G_{t+1}$ directly.)

Writing the expectation out explicitly over the four-argument $p$ (same expansion as Exercise 3.13):

$$q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_\pi(s')\big]$$

Exercise 3.20 Draw or describe the optimal state-value function for the golf example.

[Ans.] $v_{putt}(s)$ (the value of always putting) has contours near the hole labeled $-1, -2, -3,\dots$, growing outward, with a very deep dip over the sand trap since escaping it by putting alone takes many strokes. $v_*(s)$ matches $v_{putt}(s)$ exactly within putting range of the hole, since putter is already optimal there. Everywhere farther out, $v_*(s) \geq v_{putt}(s)$ and generally strictly greater: the driver covers far more distance per stroke, so locations that would take 3+ putts to hole out can be reached in 2 strokes (drive, then putt) under the optimal policy. So the $-2$ contour of $v_*$ extends much farther from the hole than the $-2$ contour of $v_{putt}$. The sand trap is still a locally low-value region under $v_*$ (an extra stroke is still needed to escape it), just less catastrophic than under $v_{putt}$.

Exercise 3.21 Draw or describe the contours of the optimal action-value function for
putting, $q_\ast(s, \text{putter})$, for the golf example.

[Ans.] $q_*(s,\text{putter})$ equals $v_{putt}(s)$ (and equals $v_*(s)$) within putting range, since committing to the putter there is already optimal. Outside putting range, $q_*(s,\text{putter})$ is the value of being *forced* to putt once from $s$ (a short move), then playing optimally afterward (switching to the driver if useful). This sits between the other two: worse than $v_*(s)$ (which would use the driver immediately, from wherever is genuinely optimal), but better than $v_{putt}(s)$ (which forces putting for every remaining stroke, not just the first). So its contours look like $v_{putt}(s)$'s contours shifted outward by roughly the distance covered in one putt, since after that first forced putt the agent recovers optimal play.

Exercise 3.22 Consider the continuing MDP shown to the
right. The only decision to be made is that in the top state,
where two actions are available, left and right. The numbers
show the rewards that are received deterministically after
each action. There are exactly two deterministic policies,
$\pi_{left}$ and $\pi_{right}$. What policy is optimal if $\gamma = 0$? If $\gamma = 0.9$?
If $\gamma = 0.5$?

*[Diagram: from the top state, `left` gives reward $+1$ then $0$ on the way back; `right` gives $0$ then $+2$ on the way back.]*

[Ans.] Each policy sends the agent around a length-2 cycle back to the top state: $\pi_{left}$ gives rewards $1,0,1,0,\dots$; $\pi_{right}$ gives rewards $0,2,0,2,\dots$. Since each reward reappears every 2 steps, both values are geometric series in $\gamma^2$:

$$v_{left}(\text{top}) = 1+\gamma(0)+\gamma^2(1)+\dots = 1\cdot(1+\gamma^2+\gamma^4+\dots) = \frac{1}{1-\gamma^2}$$

$$v_{right}(\text{top}) = 0+\gamma(2)+\gamma^2(0)+\dots = 2\gamma\cdot(1+\gamma^2+\gamma^4+\dots) = \frac{2\gamma}{1-\gamma^2}$$

Both share the same positive denominator, so comparing them reduces to comparing $1$ vs. $2\gamma$: right is optimal when $2\gamma>1 \iff \gamma>0.5$, left when $\gamma<0.5$, and they're tied at $\gamma=0.5$ (both deterministic policies achieve the same value).

- $\gamma=0$: $2\gamma=0<1 \Rightarrow \pi_{left}$ optimal.
- $\gamma=0.9$: $2\gamma=1.8>1 \Rightarrow \pi_{right}$ optimal.
- $\gamma=0.5$: $2\gamma=1 \Rightarrow$ tied, both optimal.

Exercise 3.23 Give the Bellman equation for $q_\ast$ for the recycling robot.

[Ans.] Applying $q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)[r+\gamma\max_{a'} q_*(s',a')]$ to each state-action pair, using the transition table from Exercise 3.4:

$$q_*(\text{high,search}) = \alpha\big[r_{search}+\gamma\max_{a'}q_*(\text{high},a')\big] + (1-\alpha)\big[r_{search}+\gamma\max_{a'}q_*(\text{low},a')\big]$$

$$q_*(\text{high,wait}) = r_{wait} + \gamma\max_{a'}q_*(\text{high},a')$$

$$q_*(\text{low,search}) = (1-\beta)\big[{-3}+\gamma\max_{a'}q_*(\text{high},a')\big] + \beta\big[r_{search}+\gamma\max_{a'}q_*(\text{low},a')\big]$$

$$q_*(\text{low,wait}) = r_{wait} + \gamma\max_{a'}q_*(\text{low},a')$$

$$q_*(\text{low,recharge}) = \gamma\max_{a'}q_*(\text{high},a')$$

Exercise 3.24 Figure 3.5 gives the optimal value of the best state of the gridworld as
24.4, to one decimal place. Use your knowledge of the optimal policy and (3.8) to express
this value symbolically, and then to compute it to three decimal places.

[Ans.] The best state is $A$. The optimal policy jumps immediately from $A$ to $A'$ (reward $+10$), then takes the shortest path back to $A$ — 4 steps, each with reward $0$ — before jumping again. So reward $+10$ recurs every 5 steps, giving a repeating geometric pattern:

$$v_*(A) = \sum_{k=0}^{\infty} \gamma^{5k}\cdot 10 = \frac{10}{1-\gamma^5}$$

With $\gamma=0.9$: $\gamma^5 = 0.9^5 = 0.59049$, so

$$v_*(A) = \frac{10}{1-0.59049} = \frac{10}{0.40951} \approx 24.419$$

which matches the figure's $24.4$ (to one decimal place).

Exercise 3.25 Give an equation for $v_\ast$ in terms of $q_\ast$.

[Ans.] The optimal policy is greedy — it puts all its probability on whichever action maximizes $q_*(s,a)$, so the weighted average from Exercise 3.12 ($v_\pi(s)=\sum_a \pi(a\mid s)q_\pi(s,a)$) collapses to a simple max, with no $\pi$ involved:

$$v_*(s) = \max_a q_*(s,a)$$

Exercise 3.26 Give an equation for $q_\ast$ in terms of $v_\ast$ and the four-argument $p$.

[Ans.] Same structure as Exercise 3.13, with $*$ in place of $\pi$ — this relationship only involves the environment's dynamics $p$, not any particular policy:

$$q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma\, v_*(s')\big]$$

Exercise 3.27 Give an equation for $\pi_\ast$ in terms of $q_\ast$.

Exercise 3.28 Give an equation for $\pi_\ast$ in terms of $v_\ast$ and the four-argument $p$.

Exercise 3.29 Rewrite the four Bellman equations for the four value functions ($v_\pi$, $v_\ast$, $q_\pi$,
and $q_\ast$) in terms of the three-argument function $p$ (3.4) and the two-argument function $r$
(3.5).

### Notes while reading Chapter 3

**Reward hypothesis:**
> That all of what we mean by goals and purposes can be well thought of as
> the maximization of the expected value of the cumulative sum of a received
> scalar signal (called reward). (pg 53)

> The reward signal is your way of communicating to
> the agent what you want achieved, not how you want it achieved. (pg 54)

## Chapter 4: Dynamic Programming

> DP algorithms are obtained by
> turning Bellman equations such as these into assignments, that is, into update rules for
> improving approximations of the desired value functions.

Exercise 4.1 In Example 4.1, if $\pi$ is the equiprobable random policy, what is $q_\pi(11, \texttt{down})$?
What is $q_\pi(7, \texttt{down})$?

[Ans.] Every transition is deterministic here and $\gamma=1$, so $q_\pi(s,a) = r + v_\pi(s')$ where $s'$ is wherever action $a$ leads.

$$q_\pi(11,\texttt{down}) = -1 + v_\pi(\text{terminal}) = -1 + 0 = -1$$

$$q_\pi(7,\texttt{down}) = -1 + v_\pi(11) = -1 + (-14) = -15$$

Exercise 4.2 In Example 4.1, suppose a new state 15 is added to the gridworld just below
state 13, and its actions, left, up, right, and down, take the agent to states 12, 13, 14,
and 15, respectively. Assume that the transitions from the original states are unchanged.
What, then, is $v_\pi(15)$ for the equiprobable random policy? Now suppose the dynamics of
state 13 are also changed, such that action down from state 13 takes the agent to the new
state 15. What is $v_\pi(15)$ for the equiprobable random policy in this case?

[Ans.] **First part:** state 15's actions are left$\to12$, up$\to13$, right$\to14$, down$\to$itself, and the original states are unchanged, so $v(12)=-22,v(13)=-20,v(14)=-14$ still hold. Letting $x=v_\pi(15)$:

$$x = \frac{1}{4}\big[(-1-22)+(-1-20)+(-1-14)+(-1+x)\big] = \frac{1}{4}(-60+x) \Rightarrow 3x=-60 \Rightarrow x=-20$$

**Second part:** now state 13's `down` also goes to 15 instead of looping back to itself, so $v_\pi(13)$ and $v_\pi(15)$ depend on each other. Using unchanged $v(9)=-20,v(12)=-22,v(14)=-14$, and letting $x=v_\pi(13)$, $y=v_\pi(15)$:

$$x = \frac{1}{4}\big[(-1-22)+(-1-14)+(-1-20)+(-1+y)\big] = \frac{1}{4}(-60+y) \Rightarrow y=4x+60$$

$$y = \frac{1}{4}\big[(-1-22)+(-1+x)+(-1-14)+(-1+y)\big] = \frac{1}{4}(-40+x+y) \Rightarrow x=3y+40$$

Solving simultaneously: $x=3(4x+60)+40=12x+220 \Rightarrow x=-20$, then $y=4(-20)+60=-20$. So $v_\pi(13)=-20$ (unchanged) and $v_\pi(15)=-20$ — the same value as the first part.

Exercise 4.3 What are the equations analogous to (4.3), (4.4), and (4.5), but for action-value functions instead of state-value functions?

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
so that convergence is guaranteed.

Ans.

Exercise 4.5 How would policy iteration be defined for action values? Give a complete
algorithm for computing $q_\ast$, analogous to that on page 80 for computing $v_\ast$. Please pay
special attention to this exercise, because the ideas involved will be used throughout the
rest of the book.


Exercise 4.6 Suppose you are restricted to considering only policies that are $\varepsilon$-soft,
meaning that the probability of selecting each action in each state, $s$, is at least $\varepsilon / |\mathcal{A}(s)|$.
Describe qualitatively the changes that would be required in each of the steps 3, 2, and 1,
in that order, of the policy iteration algorithm for $v_\ast$ on page 80.



Exercise 4.7 (programming) Write a program for policy iteration and re-solve Jack’s car
rental problem with the following changes. One of Jack’s employees at the first location
rides a bus home each night and lives near the second location. She is happy to shuttle
one car to the second location for free. Each additional car still costs 2 dollars, as do all cars
moved in the other direction. In addition, Jack has limited parking space at each location.
If more than 10 cars are kept overnight at a location (after any moving of cars), then an
additional cost of 4 dollars must be incurred to use a second parking lot (independent of how
many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often
occur in real problems and cannot easily be handled by optimization methods other than
dynamic programming. To check your program, first replicate the results given for the
original problem.


Exercise 4.8 Why does the optimal
policy for the gambler’s problem have such a curious form? In particular, for capital of 50
it bets it all on one flip, but for capital of 51 it does not. Why is this a good policy?



Exercise 4.9 (programming) Implement value iteration for the gambler’s problem and
solve it for $p_h = 0.25$ and $p_h = 0.55$. In programming, you may find it convenient to
introduce two dummy states corresponding to termination with capital of 0 and 100,
giving them values of 0 and 1 respectively. Show your results graphically, as in Figure 4.3.
Are your results stable as $\theta \to 0$?



Exercise 4.10 What is the analog of the value iteration update (4.10) for action values,
$q_{k+1}(s, a)$?


Note -> Chapter 4 doesn't make a lot of sense, especially page 83 (of the book). -->