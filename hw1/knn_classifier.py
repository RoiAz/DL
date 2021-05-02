import numpy as np
import torch
#from sklearn.metrics import accuracy_score 
# from sklearn.model_selection import KFold 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
        trainX, trainY = dataloader_utils.flatten(dl_train)
#         n_classes = int(y_train[0])
        classesN = trainY.unique().shape[0]
        # ========================

        self.x_train = trainX
        self.y_train = trainY
        self.n_classes = classesN
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            val, idx = torch.topk(dist_matrix[:,i], self.k, largest = False)
            class_vec = self.y_train[idx]
            y_pred[i] = torch.argmax(class_vec.bincount())
            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.
    #  - Don't use torch.cdist

    dists = None
    # ====== YOUR CODE: ======
#     diff = x1[:,np.newaxis,:] -x2
#     dists = np.sqrt(np.sum(np.square(A[:,np.newaxis,:] - B), axis=2))
# #     dists = np.sqrt((np.square(x1[:,np.newaxis]-x2).sum(axis=2)))

# #     x1SumSquare = np.sum(np.square(x1),axis=1);
#     x1SumSquare = torch.sum(x1**2,1)
# #     x2SumSquare = np.sum(np.square(x2),axis=1);
#     x2SumSquare = torch.sum(x2**2,1)

#     mul = np.dot(x1,x2.T);
#     dists = np.sqrt(x1SumSquare[:,np.newaxis]+x2SumSquare-2*mul)

    x1sum = torch.sum(x1**2,1)
#     print(x1sum.shape)
#     print(x1sum)
    x1sum.unsqueeze_(-1)
#     print(A.shape)
#     print(A)
    x1sum = x1sum.expand(x1.shape[0],x2.shape[0])
#     print(x1sum.shape)
#     print(x1sum)
    x2sum = torch.sum(x2**2,1).T
    x2sum.unsqueeze_(-1)
    x2sum = x2sum.expand(x2.shape[0],x1.shape[0]).T
    x1mulx2 = torch.matmul(x1, x2.T)
    
    dists = torch.sqrt(x1sum-2*x1mulx2+x2sum)
    # ========================

    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
#     accuracy = accuracy_score(y,y_pred)
    accuracy = torch.sum((y_pred-y) == 0).numpy()/y_pred.shape[0]
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
         ## spliting the data, not random
        # first we will see how many data we will have in eath folder
        ds_len = len(ds_train)
        fold_len = ds_len//num_folds
        
        # set a vector of how to split the data, every part will be fold_len long, and the last one will also inclode the remainder from the divider
        vec_split_lens = np.ones(num_folds) * fold_len
        vec_split_lens[-1] = ds_len - fold_len*(num_folds-1) # set the end of the vectory to inclode the remainder from the divider
        vec_split_lens = vec_split_lens.astype(np.int32)
        
        # now that we know how mush data we will have in eath folder. we will creat a list of folder idexs
        folder_id = []
        folder_start = 0
        for i in range(len(vec_split_lens)):
            folder_id.append(list(range(folder_start, folder_start+vec_split_lens[i])))
            folder_start += vec_split_lens[i]
        
        ## train
        sub_accuracie = np.zeros(num_folds)
        for i in range(num_folds):
            #get valid train
            valid = DataLoader(torch.utils.data.Subset(ds_train, folder_id[i]))
            temp = sum(folder_id[:i] + folder_id[i + 1:], []) # skip over the valid folder
            train = DataLoader(torch.utils.data.Subset(ds_train, temp))
            # train the model
            model.train(train)
            x_valid, y_valid = dataloader_utils.flatten(valid)
            sub_accuracie[i] = accuracy(y_valid, model.predict(x_valid))
            
        # append to accuracie list
        accuracies.append(sub_accuracie)
        
        """
        cv = KFold(i, True)
        for val in i:
            kf_predicts = []
            for train_index, test_index in cv.split(ds_train):
                # print("Train Index: ", train_index, "\n")
                # print("Test Index: ", test_index)

                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                classifier = ID3(X_train, X_test)
                ID3.fit(classifier.root, val)
                kf_predicts.append(classifier.predict())
            M_pred_mean = 0
            for elem in kf_predicts:
                M_pred_mean += elem / len(kf_predicts)
            accuracies.append(M_pred_mean)
        """
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
