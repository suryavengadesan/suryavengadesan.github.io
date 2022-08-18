# Imitation Learning: How well does it perform? 
#### by Surya Vengadesan
## Introduction

Imitation learning (IL) broadly encompasses any algorithm (e.g. [BC](https://ml.berkeley.edu/blog/posts/bc/)) that attempts to learn a policy for an [MDP](https://ml.berkeley.edu/blog/posts/mdps/) given expert demonstrations. In this blog post, we will be discussing how one can distinguish this task into three subclasses of IL algorithms with some mathematical rigor. In particular, we will be describing ideas presented in this recent [paper](https://arxiv.org/abs/2103.03236) that provides a taxonomixal framework for imitation learning algorithms. 

## Diving Deep

A key idea behind this paper is to define a metric that generalizes the goal of IL algorithms at large. Simply put, find a policy that minimizes the _imitation gap_ $J(\pi_E) - J(\pi)$, between our learned policy $\pi$ and an expert policy $\pi_E$. In particular, we define $J(\cdot)$, what we call a moment, which is the expection over some function. 

The goal is to then match the expert and learner moments, which the paper refers to as moment matching. In particular each type of IL algorithm can be defined by different types of moments being matched, such as reward or Q functions. Through this lens, IL can be viewed as applying optimization techniques to minimize the imitation gap. 

The paper then goes onto clearly defining the three types of IL algorithms and provides each with theoretical gaurantees, proof after proof. Specifically, the author delinates between (1) matching rewards with access to the environment (2) matching q-values with access to the environment (3) matching q-values without access to the environment. 

Now, we'll attempt to deconstruct proofs from the paper on whether an algorithm's imitation gap grows linearly or quadratically in time.

![](https://i.imgur.com/BI81QRh.jpg)

In particular, we will cover the paper's first lemma which proves a linear upper bound on the imitation gap for type (1) algorithms that moment match reward functions like [GAIL](https://arxiv.org/abs/1606.03476) or [SQIL](https://arxiv.org/abs/1905.11108). In the process, we will breakdown this craft of proving algorithmic bounds, then welcome the reader to decode the remaining proofs. 

_Lemma 1: Reward Upper Bound: If $\mathcal{F}_r$ spans $\mathcal{F}$, then for all MDPs, $\pi_E$ and $\pi \leftarrow \Psi\{\epsilon\}$, $J(\pi_E) - J(\pi) \leq O(\epsilon T)$._

That sounds like a handful; now, let's break this down.

The author precedes this lemma with general framework that formalizes the goal of imitation learning algorithms. We should look at imitation learning as a two-player game. We define the payoff function of this game with $U_1$ as follows: $$U_1(\pi, f) = \frac{1}{T}(\mathbb{E}_{\tau \sim \pi}[\Sigma_{t=1}^{T} f(s_t, a_t)] - \mathbb{E}_{\tau \sim \pi_E}[\Sigma_{t=1}^{T} f(s_t, a_t)])$$

Above, see the difference between two moments, the first moment over the sum of rewards till a time horizon T while following policy $\pi$, and the second moment over the sum of rewards till the same time horizon but following the expert policy $\pi_E$. Therefore, given some expert policy $\pi_E$, the game is to choose a policy $\pi$ that best matches the moment of the expert, while being robust to all possible MDP reward functions $f$. 

The paper defines Player 1 as the IL algorithm that attempts to select some policy $\pi$ from the set of all possible policies $\Pi \ \{\pi: \mathcal{S} \to \Delta(\mathcal{A})\}$, and whose goal is to minimize the imitation gap $U_1$. Player 2 is a discrimating reward function $f$ selected from the set of all reward functions $\mathcal{F}_r = \{\mathcal{S} \text{ x } \mathcal{A} \to [-1, 1]\}$, whose goal is to maximize the imitation gap $U_1$.   

The notion of an $\delta$-approximate solution to this game is also introduced. Given a payoff $U_j$, a pair $(\hat{f}, \hat{\pi})$ is a $\delta$-eq. solution if: $$sup_{f \in \mathcal{F}}U_j(f, \hat{\pi}) - \frac{\delta}{2} \leq U_j(\hat{f}, \hat{\pi}) \leq inf_{\pi \in \Pi} U_j(\hat{f}, \pi) + \frac{\delta}{2}$$

In addition, we define a function that magically solves this game for us, which we call the imitation game $\delta$-oracle, $\Psi\{\delta\}(\cdot)$, which takesn in a payoff function $U: \Pi \text{ x } \mathcal{F} \to [-k, k]$ and returns a (k$\delta$)-approximate equilibrium solution $(\hat{f}, \hat{\pi})$ that satifies the condition above. 

Using the tools built up, we can now prove lemma 1. First one needs to use the imitation gap, based on it's definition in the paper:  $$J(\pi_E) - J(\pi)$ $\leq sup_{f \in \mathcal{F}_r} \mathbb{E}_{\tau \sim \pi}\Sigma_{t = 1}^{T} f(s_t, a_t) - \mathbb{E}_{\tau \sim \pi_E}\Sigma_{t=1}^{T}f(s_t, a_t)$$

Next, since the sup is always positive, we can multiply by it by two and the new sup will be greater than one above. In addition, by defintion of $U_1$, you can see that term we are taking the supremum over is simply T multiplied by $U_1$. Therefore, we have:
$$J(\pi_E) - J(\pi) \leq 2T sup_{f \in \mathcal{F}} U_1(\pi, f).$$ Finally, if we apply the $\delta$-approx. equilibrium def, we can see that: 
$$sup_{f \in \mathcal{F}} U_j(f, \hat{\pi}) - \frac{\delta}{2} \leq U_j(\hat{f}, \hat{\pi}).$$ Where $\hat{f}$ and $\hat{\pi}$, the optimal values can be seen as the result of performing the minimax over the policy class $\Pi$: 
$$min_{\pi \in \Pi} max_{f \in \mathcal{F}}U_j(\pi, f)$$ which is equal to 0 since, the expert policy $\pi_E \in \Pi$, so $U_1$ (as further defined above) collapses to 0.

Finally, we have that $sup_{f \in \mathcal{F}} U_1(f, \hat{\pi})- \frac{\delta}{2} \leq 0$ and if you mulitply this inequality by $2T$, and bring the $\delta$ term to the right, you get $2T sup_{f \in \mathcal{F}}U_1(\pi, f) \leq 2T\delta$. This matches the $\epsilon$ definition from the Lemma 1, so we finally get $J(\pi_{E}) - L(\pi) \leq 2T\epsilon = O(\epsilon T)$.

The other lemma then goes onto proving that the lowerboard is also linearly constrainted.

_Lemma 2_: There exists an MDP, $\pi_E$, and $\pi \leftarrow \Psi\{\epsilon\}(U_1)$ such that $J(\pi_E) - J(\pi) \geq \Omega(\epsilon T)$

However, takes on different techiques to prove the statement, which requires an in-depth understanding of more defintions to creatively chain together multiple logical tricks. Together, lemma 1 and 2 show that for any IL algorithm that attempts to match reward moments, its imitation gap will grow linear in both the best and worst case senarios. 

The paper doesn't stop here and goes onto proving bounds for the other two taxonomixal categories, introduces a mathematical concept on MDPs called "recoverability", and proposes two new algorithms using the proven bounds, one of which is called Adversarial Reward-moment Imitation Learning that is claimed to perform with better gauranteed performance over SQIL and GAIL.  

If this material sounds interesting, check out the further readings below to get a glimpse into the world of Imitation Learning and how researchers today are incorporating rigor into the field in creative ways. 



<!--
Simple MDP Example:

$T = 2$, $\mathcal{S} = <s_1, s_2>$, $\mathcal{A} = <a_1, a_2>$

Reward 1
|   | $s_1$ | $s_2$ |
|---|---|---|
|$a_1$|1|0|
|$a_2$|0|-1|

Reward 2
|   | $s_1$ | $s_2$ |
|---|---|---|
|$a_1$|0|1|
|$a_2$|1|0|

Policy 
|   | $s_1$ | $s_2$ |
|---|---|---|
|$a_1$|0.5|0.5|
|$a_2$|0.5|0.5|

## Misc 

Intuition Left to Impart: 
- reward function versus value and q-value
- $R(s, a) = c$
- For deterministic policy 
- $V_\pi(s) = \mathbb{E}_\pi[\Sigma_{k=0}^\infty  \gamma^kR_{t+k+1}|S_t=s]$
- $Q_\pi(s, a) = \mathbb{E}_\pi[\Sigma_{k=0}^\infty \gamma^k R_{t+k+1} | S_t = s, A_t = a]$. 
- Value and Q functions are moments of the reward function without gamma and for fixed time?

Apply it to a simple MDP

Things to Add: 
-Empirically match moments are described in Graber 
-Intuitive difference between policy and reward moment matching with algorithm examples
-Show the working AdRIL algo, which relates to the bounds

Q: What knowledge does the knowledge assume the reader has when reading the paper? What knowledge would does the writer think the reader should have to get the most out of the paer? What knowledge does the writer have when writing the paper?
-->
## References

https://arxiv.org/pdf/2009.05990.pdf

http://proceedings.mlr.press/v9/ross10a.html

http://proceedings.mlr.press/v9/ross10a/ross10aSupple.pdf

https://inst.eecs.berkeley.edu/~cs188/sp21/assets/notes/note02.pdf

https://www.youtube.com/watch?v=XqgDtD_P75A&feature=emb_logo

<!-- 
https://www.johndcook.com/blog/2010/09/20/skewness-andkurtosis/


https://www.quora.com/Machine-Learning-Whats-the-intuition-behind-moment-matching-constraints-in-Maximum-Entropy-estimation 


## Misc. 
Q: How can I perform such bounds onf my own?
Q: Why does blogging just feel like summarization without provding no new insight?
Q: Connect this idea of bounds with CS170's idea of bounds?

What are the different moments you need? What is a moment? What exactly are you matching. The premise of the article is to better define how imitation learning works. 
Three contributions:
1. Unifying Moment Matching Framework
    a. MM between reward, off-Q, on-Q
    b. Off policy Q-Value moments, On-policy Q-Value Moments, On-policy reward moments
2. Imitation Learning Bounds given Problem Structure
    a. Recoverability: property between expert policy and moment class
        i. Characterizes problems where compounding errors are likely to occur
3. Construction of AdVIL AdRIL
    a. Better performance guarantees 

Definitions: 
- Advantage Function $A_t^{\pi}$
- Performance Function
- Imitation Gap: Distance from expert policy performance
- H-recoverable: A pair that satisfies from distance constraint
- Moment: Expectation over basis functions

Misc. Draft.
It next distinguishes between on-policy versus off-policy algorithms. The paper then introduces the concept of recoverability to provide both upper and lower bounds of performances for each class. These bounds are again computed throught the lens of two-player games. Finally, two algorithms are proposed with performance gaurantees offered in the previous bounds.

<!--
## Meta-reinforcement Learning

## Outline
(1) Formal Definition: 
https://arxiv.org/pdf/1611.05763.pdf
(2) Relation with Levin Search 

<!--

### Markov Decision Problems

Learning abstract task structure
Examine how meta-RL adapts to invariance in task structure

### The Two-Step Task

Learning procedure that emerges from meta-RL differs from original RL algo

### Learning abstract task structure

### One Shot Navigation
Harlow task -> visually rich environment over long time horizons
Dynamically changing sparse rewards

## Simple Principles of Meta Learning
### Intro
(1) Evaluate and compare learning methods
(2) measure the beenfits of early leanring on subsequence learning
(3) Use such evaluations to reasong about learning strategies, selecting useful one and discarding others

Scenario: 
Agent executes life long action seq in unkown env
(1) Life: t = 0 to T
(2) Learning, which modifies policy, consumes learner's life

Rethink: conventional, multiple-trail based way of measuring performance 

Basic Idea: 
(1) Learning algorithms are represented as another action
(2) Probability of learning depends on current internal state and policy
(3) These actions can modify policy and store old policy on stack to restore if needed

### MRL Cycle
(1) When t = 0, set internal state $\mathcal{I}$ 
(1a) Environmental inputs can be represented by components of I
(2) When t = 0, initialize vector $Pol$
(3) Initialize stack $S$
(4) Repeats below cycle until death


(1) Evaluate actions accordig to $Pol$ and $I$ until EVAL CRITERION satisfied
(2)



## References
Meta-RL OG
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.45.2558&rep=rep1&type=pdf
Learning to Reinforcment Learn
https://arxiv.org/pdf/1611.05763.pdf
<!--

## IRL in Finance 
## Outline or Notes:

Ill-Posed Inverse Problems
- Infinite number of solutions or no solutions at all

Behavioral Cloning
- Does not generalize well to unseen states

And I quote "methods of IRL are very useful for quantitative finance."

Use cases: "trading strategy identification, sentiment-based trading, option pricing, inference of portfolio investors, and market modeling."

"Introduce IRL methods taht can surpass a demonstrator"

## FinRL Notes:
- Action space {-1, 0, 1} = sell, hold, buy and {-k ,..., 0, ..., k} = according number of shares to sell, hold or buy
- Reward function: r(s', a, s) = v' - v, which is portfolio value at s' and s
- State space: Basically any data you can get your hands on, and feature engineering that
- Env: Dow Jones 30 stocks

## Goal

Code: Make a agent that can make money on the real market

Train PPO on GME, and use Stimulus money to make real trades. 

Long Term Goal: Pay off 4th year tuition and grad school with passive trading income? 

## Live Trading Setup

Trading Interface


## References 

FinRl Repo. https://arxiv.org/pdf/2011.09607.pdf

Elegant RL. https://github.com/AI4Finance-LLC/ElegantRL

https://www.investopedia.com/articles/active-trading/090815/picking-right-algorithmic-trading-software.asp

https://www.reddit.com/r/algotrading/comments/maxiil/is_there_an_easy_way_to_programmatically_trade/

https://developer.tdameritrade.com/apis

https://blog.quantinsti.com/algorithmic-trading-retail-traders/

https://www.mondaq.com/unitedstates/commoditiesderivativesstock-exchanges/825280/potential-regulation-of-algorithmic-trading

https://www.interactivebrokers.com/en/index.php?f=5041

https://quantra.quantinsti.com/courses

![](https://i.imgur.com/H0kCejS.png)

![](https://i.imgur.com/0zdC2ks.png)

![](https://i.imgur.com/qSTLHtZ.png)
