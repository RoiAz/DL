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
#     hypers.update({"batch_size":512, "seq_len": 64, "h_dim":256, "n_layers":3, "dropout":0.2, "learning_rate":0.001, "lr_sched_factor":0.01, "lr_sched_patience":5})
# hypers.update({"batch_size":128, "seq_len": 64, "h_dim":128, "n_layers":2, "dropout":0.2, "learning_rate":0.01, "lr_sched_factor":0.01, "lr_sched_patience":1})
    hypers.update({'batch_size': 150, 'seq_len': 64, 'h_dim': 256, 'n_layers': 3, 'dropout': 0.2, 'learn_rate': 0.001, 'lr_sched_factor': 0.3, 'lr_sched_patience': 3})
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "King"
#     temperature = 0.001
    temperature = 0.1
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences instead of training instead of training on the whole text, because of hardware limitations. It is not possible to load the entire text to the GPU. We want to acheeve compute in a reasonable amount of time.

"""

part1_q2 = r"""
**Your answer:**

It possible that the generated text clearly shows memory longer than the sequence length because every batch of characters might be affected by the hidden layers across multiple batches. Therefore, we don't really have control over the length of the generated text.


"""

part1_q3 = r"""
**Your answer:**

We don't shuffle the order of batches when training because there is a sense in the order of a text in-order to tell a story. The module remembers pass inputs and use it to try and produce new data. Shuffeling when training will ruin the order of story and memory of the module will be without true meaning.


"""

part1_q4 = r"""
**Your answer:**

1. We lower the temperature for sampling because when starting to train, the module should be filled with data, therfore default is one. But after a while when temprature gets lower, the module will switch less data. Therefore, older things from the text will be remembered.
2. If the temprature is very high, the module will frequently switch the data. So it won't remember old data, this will harm the important connection between different parts of the text.
3. If the temprature is very low, it will be hard to update memory, in that case all new parts will be according to old text and not any of the new text.


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


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

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
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
