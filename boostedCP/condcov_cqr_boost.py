import subprocess
import pandas as pd
import random
import numpy as np
import json
import os, sys
import torch
from sklearn.model_selection import KFold

from boostedCP.utils import cqr_preboost
from boostedCP.contrast_trees import contrast_pyfun, gradient_boost_condcov_contree

def condcov_cqr_boost(X_train, y_train, X_cal, y_cal,X_test, y_test, 
                      miscov_rate, seed,data_name,cache_dir,n_rounds_cv,
                      learning_rate,T_1=50,T_2=50,treesize=10,n_folds = 5, verbose = True):
    """
    Boosting the baseline CQR score for enhanced conditional coverage,
    with number of boosting rounds selected via cross validation
    
    Parameters
    ----------
    X_train, y_train, X_cal, y_cal, X_test, y_test : list, train, calibration and test features and labels
    miscov_rate: float, target miscoverage rate
    seed: integer, random seed
    data_name: string, dataset name
    cache_dir: string, cache directory
    n_rounds_cv: integer, maximum number of boosting rounds
    learning_rate: float, learning rate of gradient boosting machine for mu and sigma, 
                          which characterize a generalized Local score function
    T_1: float, temperature for sigmoid
    T_2: float, temperature for softmax
    treesize: integer, maximum number of terminal nodes in generated contrast trees
    n_folds: integer, number of cross validation folds                      
    
    Returns
    ----------
    dictionary: A dictionary containing the results of boosted conformal procedure:
        - 'conv': float, the ratio of the # of boosting rounds selected by cv wrt the maximum # of rounds n_rounds_cv
        - 'cqr_cov', 'boosted_cov': float, marginal coverage of the baseline (CQR) and boosted (CQRb) CQR procedure 
        - 'cqr_len', 'boosted_len': float, average length of CQR and CQRb
        - 'cqr_max_condcov_dev', 'boosted_max_condcov_dev': float, maximum deviation (\ell_M) from target conditional 
                                                            coverage rate of CQR and CQRb
        - 'cqr_condcov_group', 'boosted_condcov_group': list, coverage rate of each leaf node of the contrast tree
        - 'cqr_condcov_group_node', 'boosted_condcov_group_node': list, name of each leaf node assigned by the 
                                                                  contrast tree algorithm
        - 'cqr_wt', 'boosted_wt': list, # of samples in each leaf node of the contrast tree
    """
    ## Baseline evaluation: CQR conformity score function
    os.environ['print_tree'] = 'CQR'

    pred_train, pred_cal, pred_test,y_upper_cqr,y_lower_cqr = cqr_preboost(X_train, y_train, X_cal, y_cal, X_test, y_test, seed, miscov_rate)
    sys.stdout.flush() 
    
    cqr_cov = np.mean((y_test >= y_lower_cqr) & (y_test <= y_upper_cqr))
    cqr_len = np.mean(y_upper_cqr - y_lower_cqr)
    
    mu_lqr_train=(pred_train[:,0]+pred_train[:,-1])/2
    sigma_lqr_train=(pred_train[:,-1]-pred_train[:,0])/2
    mu_lqr_cal=(pred_cal[:,0]+pred_cal[:,-1])/2
    sigma_lqr_cal=(pred_cal[:,-1]-pred_cal[:,0])/2
    mu_lqr_test=(pred_test[:,0]+pred_test[:,-1])/2 
    sigma_lqr_test=(pred_test[:,-1]-pred_test[:,0])/2 
    
    json_contrast_test = contrast_pyfun(X_test,y_test,y_upper_cqr,y_lower_cqr,miscov_rate,treesize,cache_dir,data_name)
    cri_test=np.array(json_contrast_test["cri"])
    cqr_max_condcov_dev=cri_test.max()
    df = pd.DataFrame({'upper_cqr':y_upper_cqr,'lower_cqr':y_lower_cqr, 'nodex': json_contrast_test["nx"], 'y_test': y_test})
    df['ratio'] = df.apply(lambda row: (row['y_test'] >= row['lower_cqr']) * (row['y_test'] <=row['upper_cqr'] ), axis=1)
    average_ratios = df.groupby('nodex')['ratio'].mean()
    average_ratios_indexes, average_ratios_values = zip(*average_ratios.items())
    cqr_wt=json_contrast_test["wt"]
    
    ## Cross Validate for optimal number of boosting rounds
    print("\nStarting cross-validation for optimal number of boosting rounds...")    
    sys.stdout.flush() 

    os.environ['print_tree'] = 'FALSE'
    cv_loss = np.zeros((n_rounds_cv,)); 
    n_train = X_train.shape[0]; n_cal = X_cal.shape[0]; n_test = X_test.shape[0]
    kf = KFold(n_splits=n_folds)
    kf.get_n_splits(X_train)
    cv_loss = np.zeros((n_rounds_cv,)); 
    norm_fold = np.zeros((n_rounds_cv,)); 
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_train = X_train[train_index,:]
        X_train_val = X_train[test_index,:]
                                
        y_train_train = y_train[train_index]
        y_train_val = y_train[test_index]
        
        pred_train_train = pred_train[train_index,:]
        pred_train_cal = pred_train[test_index,:]
        pred_train_test = pred_train[test_index,:]
        
        mu_lqr_train_train=(pred_train_train[:,0]+pred_train_train[:,-1])/2
        sigma_lqr_train_train=(pred_train_train[:,-1]-pred_train_train[:,0])/2
        mu_lqr_train_cal=(pred_train_cal[:,0]+pred_train_cal[:,-1])/2
        sigma_lqr_train_cal=(pred_train_cal[:,-1]-pred_train_cal[:,0])/2
        mu_lqr_train_test=(pred_train_test[:,0]+pred_train_test[:,-1])/2 
        sigma_lqr_train_test=(pred_train_test[:,-1]-pred_train_test[:,0])/2 
        
        dictionary=gradient_boost_condcov_contree( X_train_train, y_train_train, X_train_val,y_train_val,
                                                     X_train_val,y_train_val,
                                                     data_name,cache_dir, T_1,T_2,
                                                  miscov_rate, n_rounds_cv,learning_rate,treesize,
                                                  mu_lqr_train_train, mu_lqr_train_cal , mu_lqr_train_test ,
                                                  sigma_lqr_train_train, sigma_lqr_train_cal , sigma_lqr_train_test, verbose = verbose)
        fold_cv_loss = np.array(dictionary["test_risk"])     
        cv_loss[:len(fold_cv_loss)] = cv_loss[:len(fold_cv_loss)] + fold_cv_loss
        norm_fold[:len(fold_cv_loss)]=norm_fold[:len(fold_cv_loss)]+1
        print(f"Fold {i} completed.")
    
    cv_loss = cv_loss/norm_fold
    best_nrounds = int(np.argmin(cv_loss))
    
    print("Cross Validation completed.")
    print("Optimal boosting rounds: ",best_nrounds)
    sys.stdout.flush() 
    ## Boost on baseline conformity score
    os.environ['print_tree'] = 'Boosted'
    if (best_nrounds>0):
        dictionary=gradient_boost_condcov_contree(X_train, y_train, X_cal, y_cal, X_test, y_test,
                                                      data_name,cache_dir, T_1,T_2,
                                                      miscov_rate, best_nrounds,learning_rate,treesize,
                                                      mu_lqr_train, mu_lqr_cal , mu_lqr_test ,
                                                      sigma_lqr_train, sigma_lqr_cal, sigma_lqr_test, verbose = verbose)
                                                      
        df = pd.DataFrame({'upper_cqr':y_upper_cqr,'lower_cqr':y_lower_cqr,'upper': dictionary["upper"], 'lower': dictionary["lower"], 'nodex': dictionary["nx"], 'y_test': dictionary["y_test"]})
        df['ratio_b'] = df.apply(lambda row: (row['y_test'] >= row['lower']) * (row['y_test'] <=row['upper'] ), axis=1)
        average_ratios_b = df.groupby('nodex')['ratio_b'].mean()
        average_ratios_b_indexes, average_ratios_b_values = zip(*average_ratios_b.items())
        
        dictionary = {k: dictionary[k] for k in dictionary if k in ('boosted_cov', 'boosted_len', 'boosted_max_condcov_dev', 'boosted_wt')}
        dictionary["boosted_condcov_group"]=average_ratios_b_values
        dictionary["boosted_condcov_group_node"]=average_ratios_b_indexes
    else:
        dictionary={
            "boosted_max_condcov_dev": cqr_max_condcov_dev,
            "boosted_wt": cqr_wt,
            "boosted_len": cqr_len,
            "boosted_cov": cqr_cov}
        
    dictionary["conv"]=best_nrounds/n_rounds_cv
    dictionary["cqr_cov"] = cqr_cov
    dictionary["cqr_len"] = cqr_len
    dictionary["cqr_max_condcov_dev"] = cqr_max_condcov_dev
    dictionary["cqr_condcov_group"]=average_ratios_values
    dictionary["cqr_condcov_group_node"]=average_ratios_indexes
    dictionary["cqr_wt"]=cqr_wt

    return dictionary


