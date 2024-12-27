import numpy as np
import torch
from typing import Tuple, Dict, List
import os, sys
import xgboost as xgb
from xgboost import XGBRegressor 
from torch.autograd import grad
from torch.autograd.functional import hessian

from boostedCP.utils import hd_wts,sigmoid,cond_miscov_dev_m,cond_miscov_dev_sm

sys.path.insert(0, os.path.abspath("../third_party/"))
from fast_soft_sort import soft_sort, soft_sort_vjp

def gradient_boost_len(X_train, y_train, X_cal, y_cal, X_test, y_test,
                       miscov_rate, n_rounds, learning_rate,
                       mu_base_train=0, mu_base_cal = 0, mu_base_test = 0,
                       sigma_base_train=1, sigma_base_cal = 1, sigma_base_test = 1, verbose=True):
    '''
    Boosting the baseline Local score for reduced interval length.
    
    Parameters
    ----------
    X_train, y_train, X_cal, y_cal, X_test, y_test : list, train, calibration and test features and labels
    miscov_rate: float, target miscoverage rate
    n_rounds: integer, total number of boosting rounds
    learning_rate: float, learning rate of gradient boosting machine for mu and sigma, 
                          which characterize a generalized Local score function
    mu_base_train, mu_base_cal, mu_base_test: list, initializtion of mu       
    sigma_base_train, sigma_base_cal, sigma_base_test: list, initializtion of sigma     
    
    Returns
    ----------
    risk: list, average interval length evaluated after each round of boosting on the training data
    risk_test: list, average interval length evaluated after each round of boosting on the test data
    mu_train, mu_cal, mu_test: list, boosted mu
    sigma_train, sigma_cal, sigma_test: list, boosted sigma
    '''
    n_train = y_train.shape[0]; n_cal = y_cal.shape[0]; n_test = y_test.shape[0]
    wts = hd_wts(n_train,miscov_rate)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dcal = xgb.DMatrix(X_cal, label=y_cal)
    dtest = xgb.DMatrix(X_test, label=y_test)
     
    mu_train=mu_base_train; mu_cal = mu_base_cal; mu_test = mu_base_test; 
    sigma_train= sigma_base_train; sigma_cal = sigma_base_cal; sigma_test = sigma_base_test; 
    
    risk = np.zeros((2*n_rounds,));    
    risk_test = []
    def coord_boost_sigma(dtrain, mu_hat, sigma_hat, verbose_sigma = True, learning_rate_sigma=1):
        def custom_loss_coord_sigma(predt: np.ndarray,
                        dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            '''
            gradients of power loss (approximated conformalised interval length) on sigma
            `\nabla_{\sigma} \ell`
            '''
            ytrue = dtrain.get_label()
            n_trained = ytrue.shape[0]
            sigma_pred = predt + sigma_hat
            score = np.abs(ytrue - mu_hat)/(np.abs(sigma_pred)+1e-6)
            qt_score=np.sum(soft_sort(score,regularization_strength=1)*wts)
            grad_Q_sc=soft_sort_vjp(values=score,vector=wts,regularization_strength=1) # gradient of soft quantile on scores
            ## gradients on sigma
            JacD_meansig_sig = np.sign(sigma_pred)/n_trained
            JacD_SC_sigma = -np.abs((ytrue - mu_hat))/((np.abs(sigma_pred)+1e-6)**2)*np.sign(sigma_pred)
            grads_sigma = 2*qt_score*JacD_meansig_sig + 2*JacD_SC_sigma*grad_Q_sc*np.mean(np.abs(sigma_pred))
            loss_hess = np.ones(grads_sigma.shape)
            return grads_sigma, loss_hess

        def custom_metric_coord_sigma(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            ''' 
            power loss (approximated conformalised interval length)
            '''
            ytrue = dtrain.get_label()
            sigma_pred = predt + sigma_hat
            score = np.abs(ytrue - mu_hat)/(np.abs(sigma_pred)+1e-6)
            qt_score=np.sum(soft_sort(score,regularization_strength=1)*wts)
            loss = np.mean(np.abs(sigma_pred))*qt_score*2
            return 'ConLen', float(loss)

        results_sigma: Dict[str, Dict[str, List[float]]] = {}
        bst_coord_sigma = xgb.train({'tree_method': 'exact', 'disable_default_eval_metric': 1, 
                                     'learning_rate': learning_rate_sigma, 'max_depth' : 1,'base_score':0},
                                    dtrain=dtrain,
                                    num_boost_round=1,
                                    obj=custom_loss_coord_sigma,
                                    custom_metric=custom_metric_coord_sigma,
                                    evals=[(dtrain, 'dtrain')],
                                    evals_result=results_sigma,
                                    verbose_eval=verbose_sigma)
        return bst_coord_sigma, results_sigma
        
    def coord_boost_mu(dtrain, mu_hat, sigma_hat, verbose_mu = True, learning_rate_mu=1):
        def custom_loss_coord_mu(predt: np.ndarray,
                                 dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            '''
            gradients of power loss (approximated conformalised interval length) on mu
            `\nabla_{\mu} \ell`
            '''
            ytrue = dtrain.get_label()
            n_trained = ytrue.shape[0]
            mu_pred = predt + mu_hat
            score = np.abs(ytrue - mu_pred)/(np.abs(sigma_hat)+1e-6)
            qt_score=np.sum(soft_sort(score,regularization_strength=1)*wts)
            grad_Q_sc=soft_sort_vjp(values=score,vector=wts,regularization_strength=1) # gradient of soft quantile on scores
            ## gradients on mu
            JacD_SC_mu=np.sign(mu_pred-ytrue)/(np.abs(sigma_hat)+1e-6) 
            grads_mu = 2*np.mean(np.abs(sigma_hat))*JacD_SC_mu*grad_Q_sc
            loss_hess = np.ones(grads_mu.shape)
            return grads_mu, loss_hess

        def custom_metric_coord_mu(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            ''' 
            power loss (approximated conformalised interval length)
            '''
            ytrue = dtrain.get_label()
            mu_pred = predt + mu_hat
            score = np.abs(ytrue - mu_pred)/(np.abs(sigma_hat)+1e-6)
            qt_score=np.sum(soft_sort(score,regularization_strength=1)*wts)
            loss = np.mean(np.abs(sigma_hat))*qt_score*2
            return 'ConLen', float(loss)

        results_mu: Dict[str, Dict[str, List[float]]] = {}
        bst_coord_mu = xgb.train({'tree_method': 'exact', 'disable_default_eval_metric': 1, 
                                  'learning_rate': learning_rate_mu, 'max_depth' : 1,'base_score':0},
                                  dtrain=dtrain,
                                  num_boost_round=1,
                                  obj=custom_loss_coord_mu,
                                  custom_metric=custom_metric_coord_mu,
                                  evals=[(dtrain, 'dtrain')],
                                  evals_result=results_mu,
                                  verbose_eval=verbose_mu)
        return bst_coord_mu, results_mu
    
   
   
    for i in range(n_rounds):        
        bst_coord_sigma, results_sigma = coord_boost_sigma(dtrain, mu_hat = mu_train, sigma_hat = sigma_train, 
                                                           verbose_sigma = verbose, learning_rate_sigma= learning_rate)
        sigma_train = sigma_train + bst_coord_sigma.predict(dtrain)
        sigma_cal = sigma_cal +  bst_coord_sigma.predict(dcal)
        sigma_test = sigma_test +  bst_coord_sigma.predict(dtest)
        risk[2*i] = results_sigma["dtrain"]["ConLen"][0]

        bst_coord_mu, results_mu = coord_boost_mu(dtrain, mu_hat = mu_train, sigma_hat = sigma_train, verbose_mu = verbose,
                                            learning_rate_mu = learning_rate)
        mu_train = mu_train + bst_coord_mu.predict(xgb.DMatrix(X_train))
        mu_cal = mu_cal +  bst_coord_mu.predict(dcal)
        mu_test = mu_test +  bst_coord_mu.predict(dtest)
        risk[2*i+1] = results_mu["dtrain"]["ConLen"][0]
        
        q_cal_adjust = np.quantile(np.abs(mu_cal-y_cal)/np.abs(sigma_cal),np.minimum(1.0, (1.0-miscov_rate)*(n_cal+1.0)/n_cal))
        upper = mu_test + q_cal_adjust*np.abs(sigma_test)
        lower = mu_test - q_cal_adjust*np.abs(sigma_test)
        cov_lb = np.mean((y_test >= lower) & (y_test <= upper))
        len_lb = np.mean(upper-lower)
         
        risk_test.append(len_lb)
        
    return risk, mu_test, sigma_test, mu_cal, sigma_cal, mu_train, sigma_train, risk_test


def gradient_boost_condcov(X_train, y_train, X_cal, y_cal, X_test, y_test, part_matrix,T_1,T_2,
                           miscov_rate, n_rounds, learning_rate,
                           mu_base_train=0, mu_base_cal = 0, mu_base_test = 0,
                           sigma_base_train=1, sigma_base_cal = 1, sigma_base_test = 1, verbose=True):
    '''
    Boosting the baseline Local score for enhanced conditional coverage.
    
    Parameters
    ----------
    X_train, y_train, X_cal, y_cal, X_test, y_test : list, train, calibration and test features and labels
    part_matrix: torch tensor, partition matrix generated by contrast trees
    T_1: float, temperature for sigmoid
    T_2: float, temperature for softmax
    miscov_rate: float, target marginal and conditional miscoverage rate
    n_rounds: integer, total number of boosting rounds
    learning_rate: float, learning rate of gradient boosting machine for mu and sigma, 
                          which characterize a generalized Local score function
    mu_base_train, mu_base_cal, mu_base_test: list, initializtion of mu       
    sigma_base_train, sigma_base_cal, sigma_base_test: list, initializtion of sigma     
    
    Returns
    ----------
    risk: list, average interval length evaluated after each round of boosting on the training data
    mu_train, mu_cal, mu_test: list, boosted mu
    sigma_train, sigma_cal, sigma_test: list, boosted sigma
    '''
    
    n_train = y_train.shape[0]; n_cal = y_cal.shape[0]; n_test = y_test.shape[0]
    wts = hd_wts(n_train,miscov_rate)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dcal = xgb.DMatrix(X_cal, label=y_cal)
    dtest = xgb.DMatrix(X_test, label=y_test)
    eps=1e-6
    mu_train=mu_base_train; mu_cal = mu_base_cal; mu_test = mu_base_test; 
    sigma_train= sigma_base_train; sigma_cal = sigma_base_cal; sigma_test = sigma_base_test; 

    risk = np.zeros((2*n_rounds,));        

    def coord_boost_sigma(dtrain, mu_hat, sigma_hat, verbose_sigma = True, learning_rate_sigma=1):
        def custom_loss_coord_sigma(predt: np.ndarray,
                        dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            '''
            gradients of max conditional coverage devaition loss on sigma
            `\nabla_{\sigma} \ell`
            '''
            ytrue = dtrain.get_label()
            n_trained = ytrue.shape[0]
            sigma_pred = predt + sigma_hat
            score = np.abs(ytrue - mu_hat)/(np.abs(sigma_pred)+eps)
            qt_score=np.sum(soft_sort(score,regularization_strength=1)*wts)
            conf_u= torch.tensor(mu_hat + qt_score*np.abs(sigma_pred), requires_grad=True)
            conf_l= torch.tensor(mu_hat - qt_score*np.abs(sigma_pred), requires_grad=True)
            cond_miscov_dev_softmax=cond_miscov_dev_sm(conf_u,conf_l,torch.tensor(y_train),part_matrix,miscov_rate,T_1,T_2, verbose=verbose)
            grad_cond_miscov_dev_conf_u=grad(cond_miscov_dev_softmax, conf_u, create_graph=True)[0].detach().numpy()
            grad_cond_miscov_dev_conf_l=grad(cond_miscov_dev_softmax, conf_l, create_graph=True)[0].detach().numpy()
            
            grad_Q_sc=soft_sort_vjp(values=score,vector=wts,regularization_strength=1) # gradient of soft quantile on scores
            JacD_SC_sigma = -np.abs((ytrue - mu_hat))/((np.abs(sigma_pred)+eps)**2)*np.sign(sigma_pred) # the diagonal 
            grad_Q_sigma = JacD_SC_sigma*grad_Q_sc
            Jac_confu_sigma = np.matmul(np.expand_dims(grad_Q_sigma,axis=1),np.expand_dims(np.abs(sigma_pred),axis=0))+qt_score*np.diag(np.sign(sigma_pred))
            Jac_confl_sigma = -Jac_confu_sigma
            
            loss_grads_sigma = np.matmul(Jac_confu_sigma,grad_cond_miscov_dev_conf_u)+np.matmul(Jac_confl_sigma,grad_cond_miscov_dev_conf_l)
            loss_hess = np.ones(loss_grads_sigma.shape)
            return loss_grads_sigma, loss_hess

        def custom_metric_coord_sigma(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            ''' 
            max conditional coverage devaition loss
            '''
            ytrue = dtrain.get_label()
            sigma_pred = predt + sigma_hat
            score = np.abs(ytrue - mu_hat)/(np.abs(sigma_pred)+eps)
            qt_score=np.sum(soft_sort(score,regularization_strength=1)*wts)
            conf_u= torch.tensor(mu_hat + qt_score*np.abs(sigma_pred))
            conf_l= torch.tensor(mu_hat - qt_score*np.abs(sigma_pred))
            cond_miscov_dev_max = cond_miscov_dev_m(conf_u,conf_l,torch.tensor(ytrue),part_matrix,miscov_rate, verbose=verbose)
            cond_miscov_dev_softmax=cond_miscov_dev_sm(conf_u,conf_l,torch.tensor(ytrue),part_matrix,miscov_rate,T_1,T_2, verbose=verbose)
            loss = cond_miscov_dev_softmax
            return 'CondCov', float(loss)

        results_sigma: Dict[str, Dict[str, List[float]]] = {}
        bst_coord_sigma = xgb.train({'tree_method': 'exact', 'disable_default_eval_metric': 1, 
                                     'learning_rate': learning_rate_sigma, 'max_depth' : 1,'base_score':0},
                                    dtrain=dtrain,
                                    num_boost_round=1,
                                    obj=custom_loss_coord_sigma,
                                    custom_metric=custom_metric_coord_sigma,
                                    evals=[(dtrain, 'dtrain')],
                                    evals_result=results_sigma,
                                    verbose_eval=verbose_sigma)
        return bst_coord_sigma, results_sigma
        
    def coord_boost_mu(dtrain, mu_hat, sigma_hat, verbose_mu = True, learning_rate_mu=1):
        def custom_loss_coord_mu(predt: np.ndarray,
                                 dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            '''
            gradients of max conditional coverage devaition loss on mu
            `\nabla_{\mu} \ell`
            '''
            ytrue = dtrain.get_label()
            n_trained = ytrue.shape[0]
            mu_pred = predt + mu_hat
            score = np.abs(ytrue - mu_pred)/(np.abs(sigma_hat)+eps)
            qt_score=np.sum(soft_sort(score,regularization_strength=1)*wts)
            conf_u= torch.tensor(mu_pred + qt_score*np.abs(sigma_hat), requires_grad=True)
            conf_l= torch.tensor(mu_pred - qt_score*np.abs(sigma_hat), requires_grad=True)
            cond_miscov_dev_softmax=cond_miscov_dev_sm(conf_u,conf_l,torch.tensor(y_train),part_matrix,miscov_rate,T_1,T_2, verbose=verbose)
            grad_cond_miscov_dev_conf_u=grad(cond_miscov_dev_softmax, conf_u, create_graph=True)[0].detach().numpy()
            grad_cond_miscov_dev_conf_l=grad(cond_miscov_dev_softmax, conf_l, create_graph=True)[0].detach().numpy()
 
            
            
            grad_Q_sc=soft_sort_vjp(values=score,vector=wts,regularization_strength=1) # gradient of soft quantile on scores
            JacD_SC_mu=np.sign(mu_pred-ytrue)/(np.abs(sigma_hat)+eps) # Diagonal of jacobian of score
            grad_Q_mu = JacD_SC_mu*grad_Q_sc
            Jac_confu_mu = np.eye(n_trained) + np.matmul(np.expand_dims(grad_Q_mu,axis=1),np.expand_dims(np.abs(sigma_hat),axis=0))
            Jac_confl_mu = np.eye(n_trained) -  np.matmul(np.expand_dims(grad_Q_mu,axis=1),np.expand_dims(np.abs(sigma_hat),axis=0))
                                   
            loss_grads_mu = np.matmul(Jac_confu_mu,grad_cond_miscov_dev_conf_u)+np.matmul(Jac_confl_mu,grad_cond_miscov_dev_conf_l)
            loss_hess = np.ones(loss_grads_mu.shape)
            return loss_grads_mu, loss_hess

        def custom_metric_coord_mu(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            ''' 
            max conditional coverage devaition loss
            '''
            ytrue = dtrain.get_label()
            mu_pred = predt + mu_hat
            score = np.abs(ytrue - mu_pred)/(np.abs(sigma_hat)+eps)
            qt_score=np.sum(soft_sort(score,regularization_strength=1)*wts)
            conf_u= torch.tensor(mu_pred + qt_score*np.abs(sigma_hat))
            conf_l= torch.tensor(mu_pred - qt_score*np.abs(sigma_hat))
            cond_miscov_dev_max = cond_miscov_dev_m(conf_u,conf_l,torch.tensor(ytrue),part_matrix,miscov_rate, verbose=verbose)
            cond_miscov_dev_softmax=cond_miscov_dev_sm(conf_u,conf_l,torch.tensor(ytrue),part_matrix,miscov_rate,T_1,T_2, verbose=verbose)
            loss = cond_miscov_dev_softmax
            return 'CondCov', float(loss)

        results_mu: Dict[str, Dict[str, List[float]]] = {}
        bst_coord_mu = xgb.train({'tree_method': 'exact', 'disable_default_eval_metric': 1, 
                                  'learning_rate': learning_rate_mu, 'max_depth' : 1,'base_score':0},
                                  dtrain=dtrain,
                                  num_boost_round=1,
                                  obj=custom_loss_coord_mu,
                                  custom_metric=custom_metric_coord_mu,
                                  evals=[(dtrain, 'dtrain')],
                                  evals_result=results_mu,
                                  verbose_eval=verbose_mu)
        return bst_coord_mu, results_mu
    for i in range(n_rounds):        
        bst_coord_sigma, results_sigma = coord_boost_sigma(dtrain, mu_hat = mu_train, sigma_hat = sigma_train, 
                                            verbose_sigma = verbose, learning_rate_sigma= learning_rate)
        sigma_train = sigma_train + bst_coord_sigma.predict(dtrain)
        sigma_cal = sigma_cal +  bst_coord_sigma.predict(dcal)
        sigma_test = sigma_test +  bst_coord_sigma.predict(dtest)
        risk[2*i] = results_sigma["dtrain"]["CondCov"][0]

        bst_coord_mu, results_mu = coord_boost_mu(dtrain, mu_hat = mu_train, sigma_hat = sigma_train, 
                                            verbose_mu = verbose, learning_rate_mu = learning_rate)
        mu_train = mu_train + bst_coord_mu.predict(xgb.DMatrix(X_train))
        mu_cal = mu_cal +  bst_coord_mu.predict(dcal)
        mu_test = mu_test +  bst_coord_mu.predict(dtest)
        risk[2*i+1] = results_mu["dtrain"]["CondCov"][0]
    return risk, mu_test, sigma_test, mu_cal, sigma_cal, mu_train, sigma_train


