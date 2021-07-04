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
**Your answer:**

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
    hypers = dict(
        batch_size=16, z_dim=32,
        data_label=1, label_noise=0.27,
        discriminator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            weight_decay=0.007,
            lr=0.0002,
            betas=(0.6, 0.7)
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            weight_decay=0.03,
            lr=0.0003,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

While training the GAN we train the discriminator and the generator, while doing that we give thee discriminator generated samples twice,
one in the genrator training and one in the discriminator training.

The discriminator get the samples as input and they are not related to his modudle so we don't need to save gradients, 
on the other hand in the genrator we want to maintain the samples gradients because the genrate process is part of model,
meaning we want to the genrator to genrate better samples.


"""

part3_q2 = r"""
**Your answer:**


1. No, the generator parmeters affected also by the discriminator parameters (minmax) as we saw in the lecture.
   In that case if the generator we stop train and the discriminator continue, the generator parameters could be bad for the model.
   
2. It mean that both genreator and the discriminator improving. Also, one can say that the two mean terms in the discriminator
   gives constant (decreases toghether) while the mean term in the generator decreases also.
"""

part3_q3 = r"""
**Your answer:**


Seems like the GAN achived better results, thats probably because the loss function. 
The main difference to our opnion is that the genrated GAN images are sharper than VAE generated images.
In VAE we use simple L2 + regularization loss function while in GAN we used the discriminative as loss function 
for every set of gamma parameters of the generator. Also, we had strong assumption on VAE model about gaussian probability.

"""

# ==============
