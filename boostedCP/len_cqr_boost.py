import numpy as np
import random
import torch
import pandas as pd
import os, sys
from sklearn.model_selection import KFold

from boostedCP.utils import cqr_preboost
from boostedCP.gradient_boost import gradient_boost_len

def len_cqr_boost(X_train, y_train, X_cal, y_cal,X_test, y_test, miscov_rate, seed, n_rounds_cv,
                             learning_rate,n_folds = 5,store=False,verbose=True):
    """
    Boosting the baseline CQR score for reduced average prediciton interval length,
    with number of boosting rounds selected via cross validation
    
    Parameters
    ----------
    X_train, y_train, X_cal, y_cal, X_test, y_test : list, train, calibration and test features and labels
    miscov_rate: float, target miscoverage rate
    seed: integer, random seed
    n_rounds_cv: integer, maximum number of boosting rounds
    learning_rate: float, learning rate of gradient boosting machine for mu and sigma, 
                          which characterize a generalized Local score function
    n_folds: integer, number of cross validation folds   
    store: boolean, whether upper and lower limits of the conformalized prediction intervals are stored
    
    Returns
    ----------
    dictionary: A dictionary containing the results of boosted conformal procedure:
        - 'conv': float, the ratio of the # of boosting rounds selected by cv wrt the maximum # of rounds n_rounds_cv
        - 'cqr_cov', 'boosted_cov': float, marginal coverage of the baseline (CQR) and boosted (CQRb) procedure 
        - 'cqr_len', 'boosted_len': float, average length of CQR and CQRb
        - 'cqr_upper', 'cqr_lower', 'boosted_upper', 'boosted_lower': (if store=TRUE) list, upper and lower limits of the 
                                                                      conformalized prediction intervals
    """          
    ## Baseline evaluation: CQR conformity score function
    pred_train, pred_cal, pred_test,y_upper_cqr,y_lower_cqr = cqr_preboost(X_train, y_train, X_cal, y_cal, X_test, y_test, seed, miscov_rate)
    cqr_cov = np.mean((y_test >= y_lower_cqr) & (y_test <= y_upper_cqr))
    cqr_len = np.mean(y_upper_cqr - y_lower_cqr)
    
    print("cqr_cov:",cqr_cov)
    print("cqr_len:",cqr_len)
    
    dictionary={"cqr_cov": cqr_cov,
                "cqr_len": cqr_len}
    if store:
        dictionary["cqr_upper"]=y_upper_cqr
        dictionary["cqr_lower"]=y_lower_cqr
        
    mu_lqr_train=(pred_train[:,0]+pred_train[:,-1])/2
    sigma_lqr_train=(pred_train[:,-1]-pred_train[:,0])/2
    mu_lqr_cal=(pred_cal[:,0]+pred_cal[:,-1])/2
    sigma_lqr_cal=(pred_cal[:,-1]-pred_cal[:,0])/2
    mu_lqr_test=(pred_test[:,0]+pred_test[:,-1])/2 
    sigma_lqr_test=(pred_test[:,-1]-pred_test[:,0])/2 
    
    ## Cross Validate for optimal number of boosting rounds
    print("Starting cross-validation for optimal number of boosting rounds...")    
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
        
        risk, mu_test, sigma_test, mu_cal, sigma_cal, mu_train, sigma_train, risk_test=gradient_boost_len(X_train_train, y_train_train, X_train_val, y_train_val, X_train_val, y_train_val,
                               miscov_rate, n_rounds_cv, learning_rate, 
                               mu_base_train=mu_lqr_train_train, mu_base_cal = mu_lqr_train_cal, mu_base_test = mu_lqr_train_test,
                               sigma_base_train=sigma_lqr_train_train, sigma_base_cal = sigma_lqr_train_cal, sigma_base_test = sigma_lqr_train_test, verbose=verbose)
        
              
        cv_loss[:len(risk_test)] = cv_loss[:len(risk_test)] + risk_test
        norm_fold[:len(risk_test)]=norm_fold[:len(risk_test)]+1
        print(f"Fold {i} completed.")
    cv_loss = cv_loss/norm_fold
    best_nrounds = int(np.argmin(cv_loss))
    
    print("Cross Validation completed.")
    print("Optimal boosting rounds: ",best_nrounds)
    
    ## Boost on baseline conformity score

    if (best_nrounds>0):
        risk, mu_test, sigma_test, mu_cal, sigma_cal, mu_train, sigma_train, risk_test=gradient_boost_len(X_train, y_train,
                               X_cal, y_cal, X_test, y_test,
                               miscov_rate, best_nrounds, learning_rate, 
                               mu_base_train=mu_lqr_train, mu_base_cal = mu_lqr_cal, mu_base_test = mu_lqr_test,
                               sigma_base_train=sigma_lqr_train, sigma_base_cal = sigma_lqr_cal, sigma_base_test =sigma_lqr_test,
                                                                                                          verbose=verbose)
        
        q_cal_adjust = np.quantile(np.abs(mu_cal-y_cal)/np.abs(sigma_cal),np.minimum(1.0, (1.0-miscov_rate)*(n_cal+1.0)/n_cal))
        upper = mu_test + q_cal_adjust*np.abs(sigma_test)
        lower = mu_test - q_cal_adjust*np.abs(sigma_test)
        cqrb_cov = np.mean((y_test >= lower) & (y_test <= upper))
        cqrb_len = np.mean(upper-lower)
        if store:
            dictionary["boosted_upper"]=upper
            dictionary["boosted_lower"]=lower
    else:
        cqrb_cov=cqr_cov
        cqrb_len=cqr_len
        if store:
            dictionary["boosted_upper"]=y_upper_cqr
            dictionary["boosted_lower"]=y_lower_cqr
    dictionary["conv"]=best_nrounds/n_rounds_cv
    dictionary["boosted_cov"]=cqrb_cov
    dictionary["boosted_len"]=cqrb_len
    print("cqr_cov:",cqr_cov)
    print("cqr_len:",cqr_len)
    print("boosted_cov:",cqrb_cov)
    print("boosted_len:",cqrb_len)
    return dictionary

        
