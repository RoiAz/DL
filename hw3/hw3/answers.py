r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q4 = r"""
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
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():

    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
#     hypers = dict(
#     batch_size=20, h_dim=100, z_dim=20, x_sigma2=0.005, learn_rate=0.1, betas=(0.7, 0.7))
    hypers = dict(
        batch_size=8,
        h_dim=100, z_dim=40, x_sigma2=0.0005,
        learn_rate=0.0005, betas=(0.8, 0.8),
    )
    # ========================
    return hypers


part2_q1 = r"""
The x_sigma2 is the variance of the likelihood, meaning how much variance we will get when to try to predict x givin z.
The larger the variance (sigma_x), the greater the scattering around the mean, 
which means that for data latents we get predictions that are more different from each other.
The more variance that we have we less demand exact fitting of the data - can help avoiding overfitting.
Also, we can interpret it as regularization strength.
"""

part2_q2 = r"""
**Your answer:**
1. The reconstruction loss means how much the reconstruction  was far from the original sample in terms of L2 norm.
   The KL divergence loss is regularization term that check how much divergence we have,
   between the posterior paramatric probability model z givin x to prior z probability (N~(0,1). less divergence smaller loss.
   
2. The  KL loss term force the latent distribution to be close to N~(0,1) distribution in our case.

3. With gaussian distribution we have simple close solution to KL loss.


"""

part2_q3 = r"""
**Your answer:**
Because we want P(x) to be close as possible to his upper bound.
Also, we had assumption that we can drop KL loss of q(ZgivinX) || p(ZgivinX) because it's small term in proportion to P(x) term.

"""

part2_q4 = r"""
**Your answer:**
When using L2 norm its easy to use log likelihood like in our case,
so instead modeling it directly and then log it we can model log of it directly.
Also, when we handling big numbers and derivatives(exploding gradient) its prefer to optimize log function.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


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
