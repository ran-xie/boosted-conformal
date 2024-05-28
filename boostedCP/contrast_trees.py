import subprocess
import pandas as pd
import random
import numpy as np
import json
import os, sys
import torch
from sklearn import preprocessing
from boostedCP.gradient_boost import gradient_boost_condcov    

def contrast_pyfun(X,y,upper,lower,miscov_rate,treesize, cache_dir,data_name):
    '''
    Calls contrast tree algorithm from R package conTree
    
    Parameter
    ---------
    X,y: data features and labels
    upper, lower: list, upper and lower limits of prediction intervals
    miscov_rate: float, target miscoverage rate
    treesize: integer, maximum number of terminal nodes in generated contrast trees
    cache_dir: string, cache directory
    data_name: dataset name
    '''
    dictionary = {
        "x_input": X.tolist(),
        "y_input": y.tolist(),
        "upper": upper.tolist(),
        "lower": lower.tolist(),
        "miscov_rate":miscov_rate,
        "treesize": treesize
    }
    
    with open(cache_dir, "w") as outfile:
        json.dump(dictionary, outfile)
    res = subprocess.call("Rscript ../boostedCP/contrast_trees_r.R", shell=True)
    
    with open(cache_dir, 'r') as openfile:
        json_contrast = json.load(openfile)
    return(json_contrast)


def gradient_boost_condcov_contree(X_train, y_train, X_cal, y_cal, X_test, y_test, data_name,cache_dir,
                                   T_1,T_2,miscov_rate, n_rounds, learning_rate,treesize=10,
                                   mu_base_train=0, mu_base_cal = 0, mu_base_test = 0,
                                   sigma_base_train=1, sigma_base_cal = 1, sigma_base_test = 1, verbose = True):
    '''
    Boosting the baseline Local score for enhanced conditional coverage. 
    In each round, we implement the contrast tree algorithm to construct the maximum deviation loss \ell_M
    
    Parameters
    ----------
    X_train, y_train, X_cal, y_cal, X_test, y_test : list, train, calibration and test features and labels
    part_matrix: torch tensor, partition matrix
    data_name: string, dataset name
    cache_dir: string, cache directory
    T_1: float, temperature for sigmoid
    T_2: float, temperature for softmax
    miscov_rate: float, target marginal and conditional miscoverage rate
    n_rounds: integer, total number of boosting rounds
    learning_rate: float, learning rate of gradient boosting machine for mu and sigma, 
                          which characterize a generalized Local score function
    treesize: integer, maximum number of terminal nodes in generated contrast trees
    mu_base_train, mu_base_cal, mu_base_test: list, initializtion of mu       
    sigma_base_train, sigma_base_cal, sigma_base_test: list, initializtion of sigma     
    
    Returns
    ----------
    risk: list, average interval length evaluated after each round of boosting on the training data
    mu_train, mu_cal, mu_test: list, boosted mu
    sigma_train, sigma_cal, sigma_test: list, boosted sigma
    '''                               
    eps=1e-6; 
    n_train = y_train.shape[0]; n_cal = y_cal.shape[0]; n_test = y_test.shape[0]
    mu_train=mu_base_train; mu_cal = mu_base_cal; mu_test = mu_base_test; 
    sigma_train= sigma_base_train; sigma_cal = sigma_base_cal; sigma_test = sigma_base_test; 
   
    train_adjust = np.quantile(np.abs(mu_train- y_train)/np.abs(sigma_train), np.minimum(1.0, (1.0-miscov_rate)*(n_train+1.0)/n_train))
    conf_u=mu_train+np.abs(sigma_train)*train_adjust
    conf_l=mu_train-np.abs(sigma_train)*train_adjust
    risk_test = []
    risk_full=[]
    
    print_tree=os.environ['print_tree'] 
    os.environ['print_tree'] = 'FALSE'

    
    for i in range(n_rounds):
        json_contrast = contrast_pyfun(X_train,y_train,conf_u,conf_l,miscov_rate,treesize,cache_dir,data_name)
        node_of_x = json_contrast["nx"]

        lb = preprocessing.LabelBinarizer()
        lb.fit(node_of_x)
        wts_nx_train=torch.from_numpy(lb.transform(node_of_x))
        wts_nx_train=wts_nx_train/torch.sum(wts_nx_train,dim=0)
        
        n_step=1
        risk, mu_test, sigma_test, mu_cal, sigma_cal, mu_train, sigma_train = gradient_boost_condcov(X_train, y_train, X_cal, y_cal, X_test, y_test, wts_nx_train,T_1,T_2,
                                   miscov_rate, n_step, learning_rate, 
                                   mu_base_train=mu_train, mu_base_cal = mu_cal, mu_base_test = mu_test,
                                   sigma_base_train=sigma_train, sigma_base_cal = sigma_cal, sigma_base_test = sigma_test, verbose=verbose)

        train_adjust = np.quantile(np.abs(mu_train- y_train)/np.abs(sigma_train+eps), np.minimum(1.0, (1.0-miscov_rate)*(n_train+1.0)/n_train))
        conf_u=mu_train+np.abs(sigma_train)*train_adjust
        conf_l=mu_train-np.abs(sigma_train)*train_adjust
        risk_full.append(risk[0])
        risk_full.append(risk[1])
        
        cal_adjust = np.quantile(np.abs(mu_cal- y_cal)/(np.abs(sigma_cal)+eps),np.minimum(1.0, (1.0-miscov_rate)*(n_cal+1.0)/n_cal))
        conf_u_test= mu_test + cal_adjust*np.abs(sigma_test)
        conf_l_test= mu_test - cal_adjust*np.abs(sigma_test)
        json_contrast_test = contrast_pyfun(X_test,y_test,conf_u_test,conf_l_test,miscov_rate,treesize,cache_dir,data_name)
        cri_test=np.array(json_contrast_test["cri"])
        risk_test.append(cri_test.max())
        
    if print_tree !='FALSE':    
        os.environ['print_tree'] = print_tree

    cal_adjust = np.quantile(np.abs(mu_cal- y_cal)/np.abs(sigma_cal), np.minimum(1.0, (1.0-miscov_rate)*(n_cal+1.0)/n_cal))
    conf_u_test=mu_test+np.abs(sigma_test)*cal_adjust
    conf_l_test=mu_test-np.abs(sigma_test)*cal_adjust
    json_contrast_test = contrast_pyfun(X_test,y_test,conf_u_test,conf_l_test,miscov_rate,treesize,cache_dir,data_name)
    cri_test=np.array(json_contrast_test["cri"])
    nx=json_contrast_test["nx"]
    boosted_max_condcov_dev=cri_test.max()
    boosted_cov = np.mean((y_test >= conf_l_test) & (y_test <= conf_u_test))
    boosted_len = np.mean(conf_u_test - conf_l_test)
    
    dictionary = {
            "x_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "upper": conf_u_test.tolist(),
            "lower": conf_l_test.tolist(),
            "nx":nx,
            "boosted_wt":json_contrast_test["wt"],
            "risk":risk_full,
            "miscov_rate":miscov_rate,
            "treesize": treesize,
            "boosted_max_condcov_dev": boosted_max_condcov_dev.tolist(),
            "boosted_len": boosted_len.tolist(),
            "boosted_cov": boosted_cov.tolist(),
            "n_update": n_rounds,
            "test_risk": risk_test
        }

    return dictionary






