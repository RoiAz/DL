r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=32, gamma=0.78, beta=0.6, learn_rate=2*1e-3, eps=1e-8, num_workers=1) #gamma - importance of future value. epsilon greedy
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.99,
              delta=0.9,
              learn_rate=1e-3,
              eps=1e-8, num_workers=1)

    # ========================
    return hp


part1_q1 = r"""
**Your answer:**
The reason there is a high variance is the scale of rewards, for example for "bad" actions we can get 20 and for "good" actions we get 50,
in which case we have increased the value function anyway.
Intuitively we want to define actions that are above average as good actions and increase their probabilty and similarly for bad actions.
For such cases we use baseline subtraction causing to be dependent only on the current state - lower variance.

"""


part1_q2 = r"""
**Your answer:**
Using Law of total expectation V function can be wrriten as expectation of Q given (s,a) for all possible (state,action) pairs from state s.
So we can see that for "good" actions the Q func value will be higher from the averge Q - V func (similar to baseline).
Meaning that if we will optimize for maximum gain (min of loss) we will learn what are the best actions to take, therefore its valid approxmation.

"""


part1_q3 = r"""
**Your answer:**
1. We can see that the graph are pretty much the same except the loss_p graph - dpg and bpg are almost costants. 
That's because the baseline subtraction causing not to see so much change due the fact the averge on the batch is close to the baseline. 
Meaning the approximate q values have low variance because all of them are close to the baseline.

we thought that the baseline methods will be smoother due the fact of smaller variance but we didn't observed that - we not sure why.

After quick research we saw that the best hyperparameters probably will be those (gamma, eps) that decreasing in time but we can't try it - we think that probably because after enough time we explore the states enough and the model should be stable - meaning the future reward should be smaller.


comparing to ACC:
we can see that most of the time the cpg is better in terms of loss, we think its because the ACC learn both Q and V functions or maybe because they have diffrents hyperparameters. 
But we can see that the ACC is better in terms of mean reward and that probably because we learn better which action will increase his immediate reward 
comparing to futrue reward.

Also we can see in loss_p graph that the ACC is not "constant" like the cpg, we think that maybe related to the fact that he also learn the "baseline" (value function) but it might be related to the assumption that the acc learn which immediate action is better comparing to future value.

"""
