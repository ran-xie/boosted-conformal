import numpy as np
import torch
from functools import partial
import pdb

import os, sys

sys.path.insert(0, os.path.abspath(".."))

from cqr import torch_models
from nonconformist.base import RegressorAdapter

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class NeuralNetworkQR:
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self, params, quantiles, verbose=False):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        quantiles : numpy array, low and high quantile levels in range (0,1)
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation

        References
        ----------
        .. [1]  Chernozhukov, Victor, Iván Fernández-Val, and Alfred Galichon.
                "Quantile and probability curves without crossing."
                Econometrica 78.3 (2010): 1093-1125.

        """

        # Store parameters
        self.params = params
        self.quantiles = quantiles
        self.verbose = verbose

        # Instantiate model
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        dropout = params['dropout']
        lr = params['lr']
        wd = params['wd']
        self.test_ratio = params['test_ratio']
        self.random_state = params['random_state']
        use_rearrangement = False
        in_shape = params['in_shape']
        hidden_size = params['hidden_size']
        learn_func = torch.optim.Adam
        self.model = torch_models.all_q_model(quantiles=self.quantiles,
                                              in_shape=in_shape,
                                              hidden_size=hidden_size,
                                              dropout=dropout)
        self.loss_func = torch_models.AllQuantileLoss(self.quantiles)
        self.learner = torch_models.LearnerOptimizedCrossing(self.model,
                                                             partial(learn_func, lr=lr, weight_decay=wd),
                                                             self.loss_func,
                                                             device=device,
                                                             test_ratio=self.test_ratio,
                                                             random_state=self.random_state,
                                                             qlow=self.quantiles[0],
                                                             qhigh=self.quantiles[-1],
                                                             use_rearrangement=use_rearrangement)

    def fit(self, X, y, cv=False):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(X, y, self.epochs, self.batch_size, verbose=self.verbose)

    def predict(self, X, quantiles=None):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX3)

        """
        return self.learner.predict(X)
