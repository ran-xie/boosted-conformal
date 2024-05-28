import subprocess
import pandas as pd
import random
import numpy as np
import json
import os, sys
import torch
from sklearn.model_selection import KFold

from boostedCP.utils import local_preboost
from boostedCP.contrast_trees import contrast_pyfun, gradient_boost_condcov_contree

def condcov_local_boost(X_train, y_train, X_cal, y_cal,X_test, y_test, 
                        miscov_rate, seed, data_name,cache_dir, n_rounds_cv,learning_rate,
                        T_1=50,T_2=50,treesize=10, n_folds = 5, verbose = True):
    """
    Boosting the baseline Local score for enhanced conditional coverage,
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
        - 'local_cov', 'boosted_cov': float, marginal coverage of the baseline (Local) and boosted (Localb) Local procedure 
        - 'local_len', 'boosted_len': float, average length of Local and Localb
        - 'local_max_condcov_dev', 'boosted_max_condcov_dev': float, maximum deviation (\ell_M) from target conditional 
                                                            coverage rate of Local and Localb
        - 'local_condcov_group', 'boosted_condcov_group': list, coverage rate of each leaf node of the contrast tree
        - 'local_condcov_group_node', 'boosted_condcov_group_node': list, name of each leaf node assigned by the 
                                                                  contrast tree algorithm
        - 'local_wt', 'boosted_wt': list, # of samples in each leaf node of the contrast tree
    """
    ## Baseline evaluation: Local conformity score function
    os.environ['print_tree'] = 'Local'
    
    preds = local_preboost(X_train, y_train, X_cal, y_cal, X_test, y_test,seed,miscov_rate)
    sys.stdout.flush() 

    mean_pred_train, mad_pred_train, mean_pred_cal, mad_pred_cal,  mean_pred_test, mad_pred_test, y_upper_local, y_lower_local = preds
    local_cov = np.mean((y_test >= y_lower_local) & (y_test <= y_upper_local))
    local_len = np.mean(y_upper_local - y_lower_local)    
    
    json_contrast_test = contrast_pyfun(X_test,y_test,y_upper_local,y_lower_local,miscov_rate,treesize,cache_dir,data_name)
    cri_test=np.array(json_contrast_test["cri"])
    local_max_condcov_dev=cri_test.max()
    local_wt=json_contrast_test["wt"]
    df = pd.DataFrame({'upper_local':y_upper_local,'lower_local':y_lower_local, 'nodex': json_contrast_test["nx"], 'y_test': y_test})
    df['ratio'] = df.apply(lambda row: (row['y_test'] >= row['lower_local']) * (row['y_test'] <=row['upper_local'] ), axis=1)
    average_ratios = df.groupby('nodex')['ratio'].mean()
    average_ratios_indexes, average_ratios_values = zip(*average_ratios.items())
    
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
                
        mean_pred_train_train = mean_pred_train[train_index]
        mad_pred_train_train = mad_pred_train[train_index]
        mean_pred_train_cal = mean_pred_train[test_index]
        mad_pred_train_cal = mad_pred_train[test_index]
        mean_pred_train_test = mean_pred_train[test_index]
        mad_pred_train_test = mad_pred_train[test_index]
        
        dictionary=gradient_boost_condcov_contree( X_train_train, y_train_train, X_train_val,y_train_val,
                                                     X_train_val,y_train_val,
                                                     data_name,cache_dir, T_1,T_2,
                                                  miscov_rate, n_rounds_cv,learning_rate,treesize,
                                                  mean_pred_train_train, mean_pred_train_cal , mean_pred_train_test ,
                                                  mad_pred_train_train, mad_pred_train_cal , mad_pred_train_test, verbose = verbose)
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
                                                      miscov_rate, best_nrounds,learning_rate, treesize,
                                                      mean_pred_train, mean_pred_cal , mean_pred_test ,
                                                      mad_pred_train, mad_pred_cal, mad_pred_test, verbose = verbose)

        df = pd.DataFrame({'upper_local':y_upper_local,'lower_local':y_lower_local,'upper': dictionary["upper"], 'lower': dictionary["lower"], 'nodex': dictionary["nx"], 'y_test': dictionary["y_test"]})
        df['ratio_b'] = df.apply(lambda row: (row['y_test'] >= row['lower']) * (row['y_test'] <=row['upper'] ), axis=1)
        average_ratios_b = df.groupby('nodex')['ratio_b'].mean()
        average_ratios_b_indexes, average_ratios_b_values = zip(*average_ratios_b.items())
        
        dictionary = {k: dictionary[k] for k in dictionary if k in ('boosted_cov', 'boosted_len', 'boosted_max_condcov_dev', 'boosted_wt')}
        dictionary["boosted_condcov_group"]=average_ratios_b_values
        dictionary["boosted_condcov_group_node"]=average_ratios_b_indexes

    else:
        dictionary={
        "boosted_max_condcov_dev": local_max_condcov_dev,
        "boosted_wt": local_wt,
            "boosted_len": local_len,
            "boosted_cov": local_cov}
        
    dictionary["conv"]=best_nrounds/n_rounds_cv
    dictionary["local_cov"] = local_cov
    dictionary["local_len"] = local_len
    dictionary["local_max_condcov_dev"] = local_max_condcov_dev
    dictionary["local_condcov_group"]=average_ratios_values
    dictionary["local_condcov_group_node"]=average_ratios_indexes
    dictionary["local_wt"]=local_wt
    return dictionary
    