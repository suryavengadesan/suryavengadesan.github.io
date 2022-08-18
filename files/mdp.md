# What are MDPs? And how can we Solve them?
#### by Surya Vengadesan
## Introduction

With deep reinforcment learning's success in beating humans at games including Go, Dota 2, and even Poker, there is a lot of excitement around the field and people want to understand its algorithms from the ground up. In this blog post, we will do just that. We will explore a fundamental building block of deep RL: Markov Decision Processess -- a framework that models how an agent behaves and learns from it's environment. In particular, we will lay out the theory of MDPs and explore approximate methods for solving them. 

### Mr. Markov Navigating College

Now, let's walk through MDPs by studying a specific agent. Specifically, an agent is an entity that interacts within its environment and interatively learns how to best act within the environment. For our example, Mr. Markov will be our agent; he is an undergraduate studying mathematics. Markov needs to make many decisions while in college (his environment). In a typical day, Markov studies for his probability theory class, sleeps in his bed, or socializes with his friends in hour long chunks. Markov is faced with a problem. Some days Markov is incredibly tired from staying up to finish his problem sets or feels stir-crazy from sitting at his desk for hours on end. Therefore, Markov needs a process to help him navigate through his daily decisions in hopes to bring him peace. 

### Modeling Mr. Markov's World

In this blog post we are going to help this poor student, by laying out a Markov Decision Process to model his world, then figure out what actions he should take in his best self-interest. Markov, at any given time, could be in one of a finite number of states. Let's denote these in a set called the state space:

$\mathcal{S}$ = {$s_1$ (studying), $s_2$ (sleeping), $s_3$ (socializing)} 

In addition, in any given state he could decide from a set of actions. Let's denote these in a set called the action space:

$\mathcal{A}$ = {$a_1$ (message a friend), $a_2$ (read Tolstoy), $a_3$ (eat a snack)}

In our model world, at every hour, Markov can perform one of these three actions from any of the three states. We shall demarcate these hours of the day by variable $t$. By taking an action at a specific state at a specific time $t$, Markov will end up in a new state $s' \in \mathcal{S}$. We shall also define a set of non-negative probabilities for each state-action pair and it's resulting action, $P(s' |s, a) \geq 0$, and with any probability distribution these must add to 1: $\Sigma_{s'} P(s'|s, a) = 1$. 

Now, we ask Markov a question: "What is your purpose? What do you value?." Markov, an aspiring future graduate student in mathematics, responds, "To one day teach probability theory to other students and perform original research within the field." To help this aspiring future probability theorist, let's add a few more definitions to his decision-making framework, that can help quantify what he values. For each state-action pair and associated next state, we assign a scalar reward $r(s'|s, a) = c$. We ask Markov a follow up question: "What's your plan? How do you plan on getting into graduate school?" He responds, "I'm going to study, study a lot." Now, to help Markov on his journey, we add a few extra formalisms. First we formalize his *plan* into a function, and call it a policy function $\pi(a|s)$ that maps every state with a specific action to take.  

To sum it up, say he's in state $s_1 = \text{studying}$, and he decided to take the action of $a_1 = \text{messaging a friend}$ according to his policy. Given $(s_1, a_1)$, he now has the following probabilities at ending up the following states after completing the action, $t(s_1 (\text{studying})| s_1, a_1)  = 0.2$ , $t(s_2 (\text{sleeping})|s_1, a_1) = 0.2$, $t(s_3 (\text{socializing})|s_1, a_1) = 0.6$, which add up to 1 as defined previously. Let's also define a set of rewards that we believe best align with what Markov says he values. We can assign real values rewards to each state he next ends up in after performing action $a_1$ in state $s_1$, $r(s_1(\text{studying})|s_1, a_1) = 10$, $r(s_2(\text{sleeping})|s_1, a_1) = 5$, and $r(s_3(\text{socializing})|a_1, s_1) = 3$. 

Together, we have incorporated stochastic behavior and measures of value and into the framework, since each state-action pair is now mapped to a probability distribution and a set of rewards that uniquely determine each next possible state. In essence, we have just laid out the mathematical formalism for MDPs, which not only helps Markov, but more generally exists as a tool that lies at the core of many control theory applications, including reinforcement learning. 

### Recap of Definitions

We have defined 4 quantities making up the tuple $<\mathcal{S},\mathcal{A}, \mathcal{T}, \mathcal{R}>$, which includes a state space, an action space, a transition function, and a reward function. In our specific example with Markov, we have 3 states, 3 rewards, and a transition and reward function that map all pairs of state-action transitions (s'|a, s) with an associated probability reward, totalling to 27 scalars each. What we have just defined above, is referred to as a *finite* MDP, because element of the tuple consists of finite sets. Below is a subset of the MDP over an arbitrary state action pair $(s_i, a_i)$.

![](https://i.imgur.com/0k1YOuf.png)


Using this structure, we would like to figure out what actions we should take to get the highest reward; fortunately, there exist proven algorithms to do exactly this. We will soon attempt to solve for an optimal policy function $\pi(\cdot)$ using said algorithms in Markov's world. Through our specific example, we are attempting to solve the old, and suprisingly relevant meme (below), but with extreme precision due to the beauty of finite MDPs. 

![](https://i.imgur.com/MQuNXU7.png)

Before, we reach the solution, however, we need a few more tools defined below.

<!---
[What is the time step for our example? Hours - incorporate that into the problem definition]

[Frame it as a discounted finite MDP]
--->


### More Fun Functions

Given these well defined transition dynamics and rewards of Markov's model world, we can define two additional functions. The first is the state-value function. This is determined by the state Markov is currently in and also his current policy function: 

$v_\pi(s) = \mathbb{E}_\pi[\Sigma_{k=0}^\infty  \gamma^kR_{t+k+1}|S_t=s]$

This can be interpreted as the long-term expected reward that Markov will receive from currently being in state $s$ and acting according to his plan. Here, we have two random variables that $S_t$ and $R_t$ that model the state and reward the agent will be in and recieve, at time step $t$, if it follows the policy defined by $\pi(\cdot)$. We have also introduced a new variable $\gamma$ which we call the discount rate, that can be any value from $0 \leq \gamma \leq 1$. The term in the expectation expands out to this form: 

$\gamma^k R_{t+k+1} = \gamma^0 R_1 + \gamma^1 R_2 + \gamma^2 R_3 + \cdots$

In essence, the value of the reward gained by taking an action and landing in a new state in the future, is being reduced by a multiple of $\gamma$ with each time step that passes. This answers the question on how much a given reward is depriciated each $t$ time step into the future, before it has been obtained.  

Altogether, the state-value function $v_\pi(\cdot)$ allows us to help Markov gauge the *value* he obtains from his tentative *plan* that we have formalized by the policy function $\pi(\cdot)$. Now, the curious Markov asks us, "Well, assume I'm working on my probability pset and decide to go read War and Peace by Leo Tolstoy, is that a good idea?" To answer this, let's add our final function to put Markov's mind at rest. Let's call it the  action-value function $q_\pi(s,a)$, which is the expected return in rewards he receives, computed similarly to the state-value function, but incorporating the reward from an initial action $a$ instead of action $\pi(s)$:

$q_\pi(s, a) = \mathbb{E}_\pi[\Sigma_{k=0}^\infty \gamma^k R_{t+k+1} | S_t = s, A_t = a]$. 

This seemingly insignificant distinction will help us when it comes time to compute the optimal policy $\pi_{*}(\cdot)$. Together, the state-value and action-value functions, are called the Bellman equations. We, will now solve for the value of these Bellman equations at each state or state-action pair, using two fundamental algorithms.

### Solving via Policy Iteration

The first main algorithm used to solve finite MDPs is called Policy Iteration. This algorithm iterates between two steps. The first step is to evaluate the value function for all states of an MDP given an arbitrary policy. This is commonly referred to as *policy evaluation*. The second step is to consider for each state, whether taking action $a \neq \pi(s)$ (not part of the policy) can improve the overall expected return. This is commonly referred to as *policy evaluation*. The algorithm loops through these steps, finding a better policy at each iteration until convergence.

<!---
The crux of this approximation method is we can iteratively find a action-value function $q(s, a)$, that is greater than or equal to the state-value function $v(s)$, which gauarantees we will converge towards finding an optimal policy. 
--->

Now, we shall implement Markov's model world using python and numpy. We follow the standard algorithm implementation outlined in Richard Sutton's RL [textbook](http://incompleteideas.net/book/first/ebook/node40.html). However, we make a few simplificiations for a quick build. (1) Our model only has 1 reward per (s', s, a) pairs, while others allow for a distribution over rewards (2) We are searching to solve for a deterministic policy, while others can solve for a stochastic policy (3) Instead of defining 56 carefully engineered rewards and transitions, we assign fixed values (4) To assure convergence of our expected long term reward (to avoid values approaching infinity), so we include a discount rate of 0.85. 

<!---
[Simplification 3: Assume discount rate is 1]

[Simplification 4: Consider time horizon of H instead of a threshold difference]
--->

We first need to make a few imports and initialize our finite MDP.

```python
import numpy as np
import matplotlib.pyplot as plt
import math

states = 3 #Number of States
actions = 3 #Number of Actions
rewards = 3 #Number of Rewards per State-action Pair
g = 0.85 #Discount Rate - gamma
v_s = [0, 0, 0] #State-value Function
pi = [0, 0, 0] #Deterministic Policy Function
theta = 1 #Hyperparameter
np.random.seed(0) #Random seed used to initalize Reward Function
R = 100 * np.random.rand(3, 3, 3) #Reward Function: intialize a 3x3x3 tensor of random values from 0 to 100
T = np.full((3,3,3), 1/3) #Transition Function: intialize a 3x3x3 tensor of 1/3s
PI = [] #List to keep track of value function for Policy Iteration
VI = [] #List to keep track of value function for Value Iteration
print(R) #Our specific Reward Function

#Randomly Initialized Reward Function
[[[0.5488135  0.71518937 0.60276338]
  [0.54488318 0.4236548  0.64589411]
  [0.43758721 0.891773   0.96366276]]

 [[0.38344152 0.79172504 0.52889492]
  [0.56804456 0.92559664 0.07103606]
  [0.0871293  0.0202184  0.83261985]]

 [[0.77815675 0.87001215 0.97861834]
  [0.79915856 0.46147936 0.78052918]
  [0.11827443 0.63992102 0.14335329]]]
```



Now for the implementation of Policy Iteration, which initializes an arbitraty policy and state-value function, then iterates between policy evaluation and policy improvement stages. 
```python
def policy_iteration(pol, val, thres):
  #Initialize Policy and Value Function
  policy = pol 
  threshold = thres
  v_s_init = val
  PI.append(v_s_init.copy())
  #Run first iteration of PE to evaluate your arbitrary policy
  value = policy_evaluation(pi, v_s_init, threshold)
  PI.append(value.copy())
  #Run first iteration of PI to find actions that might prove it
  policy_stable, policy = policy_improvement(pi, value)
  #Repeat PE and PI, until no change in policy improves performance (i.e. policy_stable = True)
  while policy_stable == False:
    value = policy_evaluation(policy, value, threshold)
    PI.append(value.copy())
    policy_stable, policy = policy_improvement(policy, value)
  return policy, value

def policy_evaluation(pol, val, thres):
  policy = pol
  threshold = thres
  v_s_init = val
  delta = math.inf
  #Evalute accuracy until Delta drops below specified threshold value
  while delta >= threshold:
    delta = 0
    #Compute expected return for each state, given the current policy
    for s in range(states):
      v = v_s_init[s]
      E_r = 0
      for s_p in range(states):
        E_r += T[s, policy[s], s_p]*(R[s, policy[s], s_p] + g*v_s_init[s_p])
      v_s_init[s] = E_r
      delta = max(delta, abs(v - v_s_init[s]))
  return v_s_init

def policy_improvement(pol, val):
  policy_stable = True
  policy = pol.copy()
  v_s_init = val
  for s in range(states):
    old_a = pol[s]
    v = v_s_init[s]
    E_r = []
    #Evalute expected value of each state and action pair
    for a in range(actions):
      e_r = 0
      for s_p in range(states):
        e_r += T[s, a, s_p]*(R[s, a, s_p] + g*v_s_init[s_p])
      E_r.append(e_r)
      #Select action the maximizes expected value
      new_a = E_r.index(max(E_r))
      #Compare if action that maximizes expected value is action specified by current policy
      if(old_a != new_a):
        #If not, modify policy and report stability to false, to ensure PI evalutes new policy
        policy[s] = new_a
        policy_stable = False
  return policy_stable, policy
#Run Algorithm
new_pi, new_v = value_iteration(pi, v_s, theta)
```
![](https://i.imgur.com/7Z9j6T0.png)
Algorithm Output: 
$v_{\pi*}(s_1, s_2, s_3)$ = [489.98445020171096, 470.622736297478, 501.629766335168]
$\pi_*(s_1, s_2, s_3)$ = [2, 0, 0]

By running this algorithm, we obtain the optimal state-value function and optimal policy shown above. Now this example is clearly idealistic, we are given the states, actions, and rewards. In the real world we don't have this, and real world problems are typically much more complicated. Figuring out how to solve more complex, obscure problems is an open challenge that the field faces to this day. In short, MDPs solve real-world problems only as best as you can model problems in the first place. Therefore, although we may not be able gaurantee that our policy will help Markov in the real-world, we hope that the theoretical implications can provide insight into his decision making. 

### Solving via Value Iteration and More

Now, let's attempt to find the optimal policy of out student's Markov's work using Value Iteration. The structure of this algorithm is very similar to Policy Iteration, but differs slightly. Instead of iterating back and forth between improvement and evaluation in Policy Iteration, Value Iteration performs a single iteration of a pseudo policy evaluation, where it additionally sweeps over the action space and perform a max operation to return the optimal state-value function and policy function upon completion. 

```python
def value_iteration(pol, val, thres):
  #Initialize Policy and Value Function
  policy = pol
  threshold = thres
  v_s_init = val
  VI.append(v_s_init.copy())
  delta = math.inf
  #Loop until delta reaches specified threshold
  while delta >= threshold:
    delta = 0
    #Evalute expected value of each state and action pair
    for s in range(states):
      v = v_s_init[s]
      E_r = []
      for a in range(actions):
        e_r = 0
        for s_p in range(states):
          e_r += T[s, a, s_p]*(R[s, a, s_p] + g*v_s_init[s_p])
        E_r.append(e_r)
      #Update policy with action that maximizes return
      policy[s] = E_r.index(max(E_r))
      #Update new state-value function accordingly
      v_s_init[s] = max(E_r)
      delta = max(delta, abs(v - v_s_init[s]))
    I.append(v_s_init.copy())
  return policy, v_s_init
#Run Algorithm
new_pi, new_v = value_iteration(pi, v_s, theta)
```
![](https://i.imgur.com/nJZINsd.png)
Algorithm Ouput:
$v_{\pi*}(s_1, s_2, s_3)$ = [490.2756261387957, 470.8914747627405, 501.8777964782847]
$\pi_*(s_1, s_2, s_3)$ = [2, 0, 0]

By running this algorithm, we obtain the optimal state-value function and optimal policy shown above. Overall, both methods follow different paths to reach the same solution, each carefully designed to use the Bellman equations to better understand the MDP. 

### Conclusion

This blog only covers the basics, and there is much more to explore in MDP theory. For example, there exists many extensions and generalizations to this framework, such as the partially observed or continous cases, studied by mathematicians and engineers alike. If this blog interested you and you want to dive deeper into algorithms for solving finite, discrete markov chains, then take a look at the references for a more detailed treatment of the topic.


### References and Future Readings
#### Key Survey Reads
Introduction in Reinforcment Learning by Sutton and Barto, Chapters 3 and 4*
Probability in Electrical Engineering and Computer Science by Jean Walrand, Chapters 11 and 12
Dynamic Programming and Markov Processes by Ronald Howard
https://en.wikipedia.org/wiki/Deep_reinforcement_learning
https://en.wikipedia.org/wiki/Markov_decision_process
http://incompleteideas.net/
*Read Bibliographical and Historial Remarks for a deeper dive

#### Related Slides and Class Sites
https://web.stanford.edu/class/cme241/lecture_slides/rich_sutton_slides/5-6-MDPs.pdf 
https://inst.eecs.berkeley.edu/~cs188/sp20/assets/lecture/lec10.pdf
https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf
https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/
https://www.cse.iitb.ac.in/~shivaram/resources/ijcai-2017-tutorial-policyiteration/tapi.pdf
https://homes.cs.washington.edu/~todorov//courses/amath579/MDP.pdf
https://www.tau.ac.il/~mansour/rl.html
https://www.cs.mcgill.ca/~dprecup/courses/AI/Lectures/ai-lecture16.pdf

#### Niche Intersecting Reads
https://en.wikipedia.org/wiki/Markov_decision_process#Extensions_and_generalizations
https://arxiv.org/pdf/1302.4971.pdf 
http://incompleteideas.net/RandomMDPs.html 

#### Talks
https://www.youtube.com/watch?v=vY-voHb22io&list=PLKlhhkvvU8-aXmPQZNYG_e-2nTd0tJE8v&index=35&t=0s

##### Other Blogs
https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da (3 part series)

<!---
### Pending Ideas
Future Ideas:
Solve Rubix Cube with Reinforcement Learning

2/14 Blog Comments
Need to redo the visualization
First Impression Charlie
Organizing the ideas better:
Start with introduction
Formalism
Value and Policy Functions
First: Different Headers for Each Section
Define the notation better, of currents state
Make clear that it is a beginner tutorial at the top
Maybe get rid of the Q stuff 
Break it up more, so that the ideas at the end make more sense
SAY: What MDPs are used for, that this is a beginner tutorial
CODE: Explain the code better
Break up the paragraphs into multiple parts 

Andre Impression
Action and state are not consistent 

Ana Cismaru
Add up the visualization before the graph, and redefine the graph
Label graph with actions and states 

Can we have code that runs in the website? HackMD: 
Display and print the graph 
What it means to solve a MDP 
And provide a short proof 
You can learn more and there is other stuff related to this.

Wednesday 2nd Draft

Saturday Final Draft 

Implement Value Iteration, Policy Iteration, LPs?
Value Iteration Done
Policy Iteration and Evaluation 

Ideas: Estimate Q and V Values, Find Optimal Policies and Optimal Value Functions with code example.

Now let's study the statistical properties and long-term behavior of this setup, in particular focusing on the idea of reward. Let's consider what markov's reward is as a student, and add it to the diagram. 

Outline:
1. Define MDP
2. Define Markov Property
2a. Define a reward
3. Define Value Function
4. Define Q Function
--->

<!---
Blogs
Ana Cismaru
-Better than last week
-Visualize the state, action, rewards
-Setups is good, but policy iteration and value iteration was confusing
Ronnie Ghose
-Dropped the college student 
-Name drop Markov
Ruchir Baronia
-substanial changes
Brian Liu
-Intuition before mathematics 
-V_pi(s) - is not explained well
Osher Lerner
-Like the metaphorical introduction
-Reword certain things 
--->




