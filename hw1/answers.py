r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers



part1_q1 = r"""
**Your answer: 

1. The test set allows us to estimate our in-sample error.
False
The in-sample error is the error rate you get on the same data-set you used to build your predictor.
The test set is responsible to indicate the out-sample error.

2. Any split of the data into two disjoint subsets would constitute an equally useful train-test split.
False
The quality of the predictor is dirrectly effected from the train set it used to practice on.
Therefore, different splits might result with different predicators.
It is intuitive to understand the claim in case of a tiny train-set, where different splits will probably lead to total diffrenet predicators.

3. The test-set should not be used during cross-validation.
True
The test-set is not responsible for chosing the hyper/parameters of the model.
Cross-validation/Training process does that.

4. After performing cross-validation, we use the validation-set performance of each fold as a proxy for the model's generalization error.
False
The generalization-error (or the out-sample error) is defined by the test-set which doesn't take part in the cross-validation process.
The performance of all folds will define the model's proxy in-sample error.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**

Your friend has trained a simple linear regression model, e.g.  𝑦̂ =\vectr𝑤𝑥⃗ +𝑏 , with some training data. He then evaluated it on a disjoint test-set and concluded that the model has over-fit the training set and therefore decided to add a regularization term  𝜆\norm𝑤⃗ 𝑤  to the loss, where  𝜆  is a hyper parameter. In order to select the value of  𝜆 , your friend re-trained the model on his training set with different values of  𝜆  and then chose the value which produced the best results on the test set.

Is your friend's approach justified? Explain why or why not.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
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
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
