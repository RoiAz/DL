r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

we have dZij/dXkm , i,k(0...127) j(0...2047), m(0...1023)
the shape of  Jacobian will be 4D - (128,2048,128,1024)

for each Zij we have 128*1024 Xkm
we have 128*2048 Zij
so we will need 128*1024*128*2048*4b = 128Gb

"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.001
    lr = 0.01
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.5,  0.05,  0.005,  0.0003, 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr, = 0.01, 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
section 1:
yes, we can see that when we have dropout =0 we have overfit because we get low loss on train set and high loss on test set.
On the other hand, when dropout !=0 we can see the opposite, the dropout here help to prevent overfit.

section 2:
High droput may cause undefit, in that case we will get lower accuracy than mid dropout. 
Also, if we sace the only the relevant parameters, the memory that high dropout require is lower than low dropout because we have less parmaters to train.

"""

part2_q2 = r"""
**Your answer:**
it's possible because the accuracy is according to number of right predications while in loss we sum the total "distance" so we can get more right predictions and it the same time to increase the "distances" of wrong predictions.
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
(1).
The number of parameters in each layer should be calculated by K(C_in*F^2 +1)
With the bottleneck technique(Assuming no bias - negligible):
CONV1 = 64*(256*1^2)=64*256
CONV2 = 64*(64*3^2)= 64*64*9
CONV3 = 256*(64*1^2)= 64*256
total = 64*(256+576+256) = 6,528

Without the bottleneck technique(Assuming no bias):
CONV1 = 256*(256*3^2)=256*256*9
CONV2 =256*(256*3^2)= 256*256*9
total = 256*256*18 = 589,824

The regular block uses ~90times more parameters than the bottleneck block 
we would like to preserve the model accuracy on one hand, yet on the other hand quantize the model into the smallest number of levels possible. The impact of quantization is twofold. First, it allows us to store the model with much fewer bits, i.e., less memory. Second, it helps us replace expensive operations with much cheaper operations.

(2).
The number of floating point operations in each layer will be quite simpler to estimate by thinking about the output of each layer.
We have K channels out of maps, where each index in the h*w map was computed by Cin*F^2 floating operations.
Hence:(C_in*F^2)*(K*h*w)_out
Therefore we'll have to multiply the number of parameters extracted in each layer by the dimensions in order to get an estimation for the amount of the floating point operations.

With the bottleneck technique(Assuming no bias - negligible).
Dimensions(h,w) relative to hyper-parameters P,D,S,F , we will assume P=0,D=1,S=1 for simplicity
CONV1 = (256*1^2)*(64*h1*w1) = 256*64*h1*w2
CONV2 = (64*3^2)*(64*h2*w2)= 64*64*9*(h1-2)*(h2-2) 
CONV3 = (64*1^2)*(256*h3*w3)=64*256*(h1-2)*(h2-2) 
total = CONV1 + CONV2 + CONV3 ~ (h1*w1)(256*64+64*64*9+64*256) ~ (h1*w1)*69,632

Without the bottleneck technique(Assuming no bias):
CONV1 = (256*3^2)(256*(h1-2)*(w1-2))= 256*256*9*(h1-2)*(w2-2) ~ 589,824(h1*w1)
CONV2 = (256*3^2)(256*(h1-4)*(w1-4))= 256*256*9*(h1-4)*(w2-4) ~ 589,824(h1*w1)
total = CONV1 + CONV2 ~ 1,179,648(h1*w1)

The regular block uses above ~15times more floating point operations than the bottleneck block 

(3). 
Lets take for example the cifar10 dimensions as an input, where each pic has a 32X32 dimensions.
In the regular block through the main path we will get an output of 28X28X256 where in the bottleneck block we will get 30X30X256
In both cases we combine the input of 32X32X256

The differences are:
    (1) spatially (within feature maps) - 
    (2) across feature maps - 



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**

(1).
In deeper networks we can be much more efficient in terms of computation and number of parameters. Deeper networks are able to create deep representations, at every layer, the network learns a new, more abstract representation of the input.
With that said, increasing depth beyond a critical threshold might lead to a decrease in test accuracy.

Disclaimer - our answers reply on those specific tests over a particular train and test sets, results might be different and we believe they likely to be different over a larger train/test sets.

In general our expectation were to get better results in deeper cnn for accuracy and loss on train set while we took into account the potential "over-fitting" on test set. In practice things were more complicated and different.

L8 train and test accuracies were good enough but a closer look will suggest a potential "over-fitting" for the model, while the test accuracy graph is straggling and not consistent(V shapes). We can also notice a train loss that doesn't improving over time.

L2 and L4 were depth tests with similar results and the best between all four. While both of them got high train and test accuracies, the loss of L2 was quit better in both the train and test sets. As we mentioned before it didn't match exactly with our expectation, the assumption was that L2 is probably too shallow leading to underfit where in practice L2 was able to train good enough and stay more generalized with good predication compare to L4 (again, the differences are small).
We believe both L2 and L4 although "shallow" were able to extract necessary features and yet stay generalized enough leading to good predications and low loss.

q2 - Were there values of L for which the network wasn't trainable? what causes this? Suggest two things which may be done to resolve it at least partially.
(2).
At first at L16 the network wasn't trainable, we got 0,0 dimension error, which occurred due to some dimensions shrinking until they reach 0.
Two suggested option to solve that:
The first and the action we took was to make sure beforehand that the network is not going to work with parameters that are too small in the beginning, where there will be a chance that they will decrease and will be considered to be 0.
Another option is to make sure that for all of the parameters in the network, if they go below some threshold we will manually multiply them by a certain constant.

Later L16 network wasn't trainable, reach poor results, without improving over time.
We have an assumptions regrading the reason causing this result:
The deeper the model, the more small changes may be amplified farther deeper.
This process might cause a 'Covariate shift', which happens due to differences in the input (batch) distributions compare to training distribution, (without any change in the underlying mapping from inputs to outputs).
We also suggest a potential solution:
Using Batch Normalization would scale the outputs and is known to reduce internal covariate shift, while also help with granting faster learning rate.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**

The bigger the value for K, the more local features we can learn to the model.
This is also why we can notice an "over-fitting" around large K's, where the train accuracy is fitting realy fast really high and the results for the test accuracy are just the opposite. The same way for the train loss wich degrade with iteration, but on the test set it is rising up with iterations. In this mannar it is similar enough to results of expirement1.1


than the optimal K values for L2 (which seems to be K32 by a large margin).


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**
The deeper the network, the more it is hard to train, we can notice pretty clearly the train accuracies L1,L2,L3,L4 from the best to worst (respectively). Same for the test accuracy.
L1 seems to be slightly more capable of higher accuracy than L2

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q6 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
