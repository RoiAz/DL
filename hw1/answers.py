r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers



part1_q1 = r"""
**Your answer:**

1.**False**
The in-sample error is the error rate you get on the same data-set you used to build your predictor.
The test set is responsible to indicate the out-sample error.

2.**False**
The quality of the predictor is dirrectly effected from the train set it used to practice on.
Therefore, different splits might result with different predicators.
It is intuitive to understand the claim in case of a tiny train-set, where different splits will probably lead to total diffrenet predicators.

3.**True**
The test-set is not responsible for chosing the hyper/parameters of the model.
Cross-validation/Training process does that.

4.**False**
The generalization-error (or the out-sample error) is defined by the test-set which doesn't take part in the cross-validation process.
The performance of all folds will define the model's proxy in-sample error.


"""

part1_q2 = r"""
**Your answer:**

**My friend's approach is wrong**
You don't supose to choose hyper-parameters based on test-set.Because, it results with over-fitting on the test-set.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The learning procces of the weights is accordingly to the delta value, meaning that for every delta value the weights will change accordingly so that
the loss function value will be aproximaly the same.

"""

part3_q2 = r"""
**Your answer:**
1. The linear model learns the weights of each class, in this case every class represent digit 0-9. For each class we have the same number of weights as the picture size.
   The erros that we see derive from the fact that some calls have better score for specific prediction.
   For example, we can see the wrong preditiction on the digit 9 that in this case is similar to 4, also those two classes have similar shape meaning the weights will get 
   high value in similar ares.
   
2. This method is diffrent from KNN because in this approch we try to learn the feature of every digit trough the weights however in KNN we just measure the "distance" between the rest of the set and predict accordingly to that.

"""

part3_q3 = r"""
**Your answer:**
We think that our learning rate is good:
* Is not too high because it seems like we have right convergence - if that was the case we should expect higher loss.
* Is not too small because it seems like its converge prrety fast - if that was the case we should expect slower convergence rate but maybe with lower loss.

we think that our model is slightly overfit because we got good accuracy on the test set.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
The ideal pattern we want to see is test-set predictions with error 0 - meaining located on y-y.pred=0 line.
This scenario is not likley possible, so we will want similar result on the train-set but not exactly the same to avoid overfit.
We can see the plot above have pretty god generalization be its possible to do better.
From the comparing we can see that we got better results after the K-fold process - more instances are close to error 0.

"""

part4_q2 = r"""
**Your answer:**
1. Yes - The model is stil lineare because the parmaters still linear.
    for examlpe, we can mark the feature $x1^{2} = x'$ and as you can see its still linear
    
2. Theoretically we can, but a great classifier might be a complex relationship between the features.

3. With non-linear features we create another dimensions where the linear classifier isn't enough for their representation, the separation is still linear thatfore its still hyperplane

"""

part4_q3 = r"""
**Your answer:**
1. In log space the 𝜆 values are samller from linear space, if we go linear 𝜆 will get bigg values and the weights will have small values as result. In the extreme case the weights will be 0 and the model will have underfitting (higher 𝜆 less complex model).

2. len(degree_range) * len(lambda_range) * k_fold = 3 * 20 * 3 = 180

"""

# ==============
