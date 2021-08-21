r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers



def v_gan_hyperparams():
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
    type = 'Adam'
    hypers = dict(
        batch_size=32,
        z_dim=128,
        data_label=1,
        label_noise=0.2,
        discriminator_optimizer=dict(
            type="SGD",
            lr=0.0075 ,
        ),
        generator_optimizer=dict(
            type=type,
            lr= 0.001,
            betas =(0.5, 0.999),
        ),
    )
    # ========================
    return hypers

def sn_gan_hyperparams():
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
    type = 'Adam'
    hypers = dict(
        batch_size=32,
        z_dim=128,
        data_label=1,
        label_noise=0.2,
        discriminator_optimizer=dict(
            type="SGD",
            lr=0.0075 ,
        ),
        generator_optimizer=dict(
            type=type,
            lr= 0.001,
            betas =(0.5, 0.999),
        ),
    )
    # ========================
    return hypers

def w_gan_hyperparams():
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
    type = 'RMSprop'
    hypers = dict(
        batch_size=32,
        z_dim=128,
        data_label=1,
        label_noise=0.0002,
        discriminator_optimizer=dict(
            type=type,
            lr=0.000035 ,
        ),
        generator_optimizer=dict(
            type=type,
            lr=0.0001 ,
        ),
        N = 5 # for each generator update, we will to 5 discriminator updates.
    )
    # ========================
    return hypers
