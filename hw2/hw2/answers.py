r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

we have dZ_i,j/dX_k,m , i,k(0...127) j(0...2047), m(0...1023)
the shape of  Jacobian will be 4D - (128,2048,128,1024) 

for each Z_i,j we have 128x1024 X_k,m 
we have 128x2048 Z_i,j 
so we will need 128x1024x128x2048x4b = 128Gb 

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
For example looking at the accuracy of the train we can watch the blue line(droput=0) with the highest result(above the green and orange lines), where in the test results the blue line is at the bottom.


section 2:
Dropout is a regularization approach intended to mitigate overfitting through simulating sparse activations from the network.
Comparing the accuracy of train and test of green(dropout=0.8) and orange(dropout=0.4):Watching the accuracy of the train we get the green(dropout=0.8) with better results while watching at the accuracy of the test we get the orange(dropout=0.4) with highest results, we believe that is because the orange was able to extract more information out of the inputs, the dropout was too agressive in the case of the green.
Comparing the loss of train and test of green(dropout=0.8) and orange(dropout=0.4): we can notice lower(better) loss for the green(dropout=0.8) in the train loss. While in the test loss we learn similar results for the green and orange dropouts, when the green dropout is more stable.
The models' tendency to over-fit the data is evident by the significant performance-disparity between training and testing. In our case, over-fitting is likely caused by the network being shallow and wide (large number of parameters and few layers). 


"""

part2_q2 = r"""
**Your answer:**
It's possible.
The accuracy measures how many samples were correctly classified, while the loss measures how far were we, the distance, from the right predication.
For example, we could make an improvement in terms of accuracy by getting more samples correctly, but at the same time increase our probability for a wrong predication on other samples in such a way that overall increased our loss function. Therefore, it is possible for the test loss to increase for a few epochs in conjunction with an increase in test accuracy.

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
In the regualr block(left block) our kernel will be with the 3x3x256 dimensions while in the bottleneck block(right clock) the kernel will be 1x1x256 dimensions.
As showed in class, within feature maps means over the spatial extent. Across feature maps means across different channels.
Therefore, in terms of the difference in ability to combine the input spatially(within feature maps) we can claim the regular blcok (left block) is more spatially involved while both of the blocks are combining the 256dim depth(of channels), there for are the same in the manner of across feature maps.
We can also claim that the left block combines more of the input in every output result because of it's bigger kernel(depends on the meaning of "difference in ability to combine the input").


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

(2).
At first at L16 the network wasn't trainable, we got 0,0 dimension error, which occurred due to some dimensions shrinking until they reach 0 (Pooling layers too often).
Two suggested option to solve that:
The first and the action we took was to make sure beforehand that the network is not going to work with parameters that are too small in the beginning, where there will be a chance that they will decrease and will be considered to be 0.
Another option is to make sure that for all of the parameters in the network, if they go below some threshold we will manually multiply them by a certain constant.

Later L16 network wasn't trainable, reach poor results, without improving over time.
We have an assumptions regrading the reason causing this result:
The reason might be due to the regularization process where the weights are too small close to zero and no learning process is acquired.
A possible solution is to decrease the regularization coefficient.

The deeper the model, the more small changes may be amplified farther deeper.
This process might cause a 'Covariate shift', which happens due to differences in the input (batch) distributions compare to training distribution, (without any change in the underlying mapping from inputs to outputs).
We also suggest a potential solution:
Using Batch Normalization would scale the outputs and is known to reduce internal covariate shift, while also help with granting faster learning rate.

"""

part3_q3 = r"""
**Your answer:**
Looking in the graphs at the test accuracies we can notice that the best results are coming from Ks 32 64. In the L2 depth we get the best result for K 32 and with the higher the depth is, the more K 64 is better. For L 4 the results are pretty similar for 32 and 64 features. For L 8 the K 64 is quit better than the K 32. We assume that the net can learn the features better in depth leading to improvement in test accuracy.

In the exp1.1 the graphs are built with same number of features and the comparison is between the different depths.
In the exp1.2 the graphs are built with same depth and the comparison is between the different number of features.
In both we ran over L 2 4 8 and K 32 64, the difference in this exp1.2 is the addition of K 128 256(with no L-16 depth).

The bigger the value for K(the amount of features), the more local features we can learn to the model.
This is also why we can notice an "over-fitting" around large K's, where the train accuracy is fitting realy fast really high and the results for the test accuracy are just the opposite(we can notice the orange and blue with the lowest test accuracies in most L-depths). The same way for the train loss wich degrade with iteration, but on the test set it is rising up with iterations. In this mannar it is similar enough to results of expirement1.1


"""

part3_q4 = r"""
**Your answer:**
The deeper the network is, in respect to a const number of epochs, the lower the accuracies.
L1,L2,L3,L4 from the best to worst(respectively).
Same for the test accuracy. L1 seems to be slightly more capable of higher accuracy than L2.
We can notice that adding more consecutive layers with the same amount of features results with lower test accuracies. We believe it is due to an over-fitting progress.

"""

part3_q5 = r"""
**Your answer:**
Looking at the graphs for the fixed K32 with different depths, L 8 16 32
We see that the L16_K32 and L32_K32 stopped to learn early while L8 continued (might cause overfitting to train set). L8 accuracy on the train set reached the highest number and the loss on train set was the ninimum between all depths. 
For the test accuracy and loss L8 depth achieved bad results compare to other depths.
The main difference in results compare to exp1.1 is that we see that no matter how deep the model was, it was still trainable!
The skip connection does seem to solve the issue we've encountered in exp1.1, likely due to the sum of the shortcut path that joins the main path though a ReLU.


Looking at the graphs for the fixed K 64 128 256 with the different depths, L 2 4 8.
The introduction of skip connections via Residual Networks produced results getting better with increasing depth, with more than 60% accuracy for the L8 depth network.
Those results are just the opposite from the results from exp1.3 where we got lower results with increasing of net's depth. Probably because of the skip connection which helps to reduce over-fitting in deeper networks.

"""

part3_q6 = r"""
**Your answer:**

q1 - Explain your modifications to the architecture which you implemented in the YourCodeNet class.
(1). 
Our architecture is using the introduction of skip connections via Residual Networks in order to handle the problem of vanishing gradients in a deep network.
We used Batch Normalization followed by activation layer after every convolution inside the residual blocks, we didn't use dropouts inside the main path of every block(dropouts=0) but we did use MaxPool2d followed by a dropout layer towards the end of our network: this is meant to assist with trainability in higher depths, and help solve the issues we were seeing in the previous experiments(We tried several architectures until we finally came up with this one yielding the better results).

q2 - Analyze the results of experiment 2. Compare to experiment 1.
(2).
In overview our networks results are 20% higher than the best results achieved in expirements 1.

Say that we look online for solutions to vanishing gradient problem and used residual blocks to help us. 
Why we added Batch Normalization
Why we moved the dropouts in the residual
Write something about L3 getting best results (early-stopping)


"""
# ==============
