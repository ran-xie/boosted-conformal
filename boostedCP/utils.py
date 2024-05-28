import numpy as np
import pandas as pd
import random
import torch
from sklearn.ensemble import RandomForestRegressor
import pdb
import os, sys
from scipy.stats import beta
from sklearn.tree import DecisionTreeRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from boostedCP.getData import get_real_data

sys.path.insert(0, os.path.abspath("../third_party/"))
from cqr_comparison import ConformalizedQR, NeuralNetworkQR

def local_preboost(X_train, y_train, X_cal, y_cal, X_test, y_test,seed,miscov_rate):
    """ 
    Train model to obtain mean and mad estimators and compute 
    conformalized prediction intervals produced by the baseline Local conformity score  
    """
    # parameters of random forests    
    n_estimators = 1000
    min_samples_leaf = 40
    max_features = 1
    random_state = seed
    
    print(f"Fitting RF model with {X_train.shape[0]} observations...")

    # define the conditonal mean estimator as random forests (used to predict the labels)
    mean_estimator = RandomForestRegressor(n_estimators=n_estimators,
                                           min_samples_leaf=min_samples_leaf,
                                           max_features=max_features,
                                           random_state=random_state)

    # define the MAD estimator as random forests (used to scale the absolute residuals)
    mad_estimator = RandomForestRegressor(n_estimators=n_estimators,
                                          min_samples_leaf=min_samples_leaf,
                                          max_features=max_features,
                                          random_state=random_state)
    
    # fitting mean and mad
    mean_estimator.fit(X_train,y_train)
    residual_prediction = mean_estimator.predict(X_train)
    residual_error = np.abs(residual_prediction- y_train)
    log_err = residual_error
    mad_estimator.fit(X_train, log_err)
    
    # predicting
    mean_pred_train = residual_prediction
    mean_pred_cal = mean_estimator.predict(X_cal)
    mean_pred_test = mean_estimator.predict(X_test)
    
    mad_pred_train = mad_estimator.predict(X_train)
    mad_pred_cal = mad_estimator.predict(X_cal)
    mad_pred_test = mad_estimator.predict(X_test)
    
    n_cal = y_cal.shape[0]
    cal_adjust = np.quantile(np.abs(mean_pred_cal- y_cal)/np.abs(mad_pred_cal), np.minimum(1.0, (1.0-miscov_rate)*(n_cal+1.0)/n_cal))
    y_lower_local = mean_pred_test - cal_adjust*np.abs(mad_pred_test)
    y_upper_local = mean_pred_test + cal_adjust*np.abs(mad_pred_test)
    return mean_pred_train, mad_pred_train, mean_pred_cal, mad_pred_cal,  mean_pred_test, mad_pred_test, y_upper_local, y_lower_local



def cqr_preboost(X_train, y_train, X_cal, y_cal, X_test, y_test, seed, miscov_rate):
    """ 
    Train model to obtain quantile estimators and compute 
    conformalized prediction intervals produced by the baseline CQR conformity score  
    """
    # Quantiles
    quantiles = [miscov_rate/2, 1-miscov_rate/2]
    
    level = "fixed"

    # Quantiles for training
    if level == "cv":
        quantiles_net = [miscov_rate, 0.5, 1-miscov_rate]
    else:
        quantiles_net = [miscov_rate/2, 0.5, 1-miscov_rate/2]

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize cuda if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ## Unpack Data
    n_train = y_train.shape[0]
    n_cal = y_cal.shape[0]

    X_train = np.concatenate((X_train,X_cal), axis=0)
    y_train = np.concatenate((y_train,y_cal), axis=0)

    # Reshape the data
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    idx = np.arange(n_train+n_cal)
    idx_train, idx_cal = idx[:n_train], idx[n_train:(n_train+n_cal)] 
    
    # Parameters of the neural network
    params = dict()
    params['in_shape'] = X_train.shape[1]
    params['epochs'] = 1000
    params['lr'] = 0.0005
    params['hidden_size'] = 64
    params['batch_size'] = 64
    params['dropout'] = 0.1
    params['wd'] = 1e-6
    params['test_ratio'] = 0.05
    params['random_state'] = seed
    # Initialize neural network regressor
    model = NeuralNetworkQR(params, quantiles_net, verbose=False)
    # Initialize regressor for hyperparameter tuning
    model_tuning = NeuralNetworkQR(params, quantiles, verbose=False)

    cqr = ConformalizedQR(model, model_tuning, X_train, y_train, idx_train, idx_cal, miscov_rate)

    pred_train, pred_cal, pred_test = cqr.predicted_quantiles(X_test, y_test, miscov_rate)
   
    # Compute CQR intervals
    lower, upper = cqr.predict(X_test, y_test, miscov_rate, method = "CQR")
    sys.stdout.flush()
    
    return pred_train, pred_cal, pred_test, upper,lower

def hd_wts(n,miscov_rate):
    """
    Harrel-Davis quantile estimator weights
    """
    a=(1-miscov_rate)*(n+1);b=miscov_rate*(n+1)
    wts_hd=np.zeros((n,))
    for i in range(1,n+1):
        wts_hd[i-1]= beta.cdf(i/n, a, b) - beta.cdf((i-1)/n, a, b)   
    return wts_hd

def sigmoid(z,T=50):
    """
    Sigmoid function
    """
    res = 1/(1+torch.exp(-T*z))
    return res  

def cond_miscov_dev_sm(conf_u,conf_l,y,part_matrix,miscov_rate,T_1=50,T_2=50,verbose=True):
    '''
    Given a conformalized prediction interval and a partition of the feature space, 
    computes the soft maximum of conditional coverage deviation from the target rate
    
    Parameters
    ----------
    conf_u, conf_l: torch tensor, upper and lower limits of prediction interval
    y: torch tensor, labels
    part_matrix: torch tensor, partition matrix
    miscov_rate: numeric, target miscovrage rate
    T_1: temperature for sigmoid
    T_2: temperature for softmax
    '''
    group_miscov_instance=part_matrix*(1-sigmoid(conf_u-y,T=T_1).unsqueeze(1)*sigmoid(y-conf_l,T=T_1).unsqueeze(1))
    group_miscov=torch.sum(group_miscov_instance,dim=0)
    group_miscov_dev=torch.abs(group_miscov-miscov_rate)
    cond_miscov_dev_softmax=torch.log(torch.sum(torch.exp(T_2*group_miscov_dev)))/T_2
    if verbose:
        print("cond_miscov_dev_softmax:",cond_miscov_dev_softmax)
    return cond_miscov_dev_softmax

def cond_miscov_dev_m(conf_u,conf_l,y,part_matrix,miscov_rate, verbose = True):
    '''
    Given a conformalized prediction interval and a partition of the feature space, 
    computes the maximum of conditional coverage deviation from the target rate
    
    Parameters
    ----------
    conf_u, conf_l: torch tensor, upper and lower limits of prediction interval
    y: torch tensor, labels
    part_matrix: torch tensor, partition matrix
    miscov_rate: numeric, target miscovrage rate
    '''
    group_miscov_instance=part_matrix*(1-(conf_u>=y).long().unsqueeze(1)*(y>=conf_l).long().unsqueeze(1))
    group_miscov=torch.sum(group_miscov_instance,dim=0)
    group_miscov_dev=torch.abs(group_miscov-miscov_rate)
    cond_miscov_dev_max=group_miscov_dev.max()
    if verbose:
        print("group_miscov_dev.max:",group_miscov_dev.max())
    return cond_miscov_dev_max
    


def rounding(numbers):
    """
    Round percentages to integers.
    """
    numbers=np.array(numbers)
    numbers=numbers/np.sum(numbers)*100
    numbers=numbers.tolist()
    rounded_numbers = [round(num) for num in numbers]

    sum_rounded = sum(rounded_numbers)
    difference = 100 - sum_rounded

        sorted_indices = sorted(range(len(numbers)), key=lambda i: -abs(numbers[i] - rounded_numbers[i]))

    adjusted_numbers = rounded_numbers.copy()
    for i in range(abs(difference)):
        index = sorted_indices[i]
        adjusted_numbers[index] += 1 if difference > 0 else -1
    return adjusted_numbers


def plot_len(data_name,ratio,seed,results_local,results_cqr,
             miscov_rate=0.1,margin=dict(l=0, r=0, t=90, b=70),y_sub=1.01,path=False,scale=False):
    """
    Plot Figure 4
    """
    df_local,df_boosted_local,df_cqr,df_boosted_cqr = plot_len_data_process(data_name,ratio,seed,results_local,results_cqr,miscov_rate)
    bar_colors = ['#6baed6', '#fd8d3c', '#31a354']
    fig = make_subplots(rows=1, cols=4,column_widths=[0.3, 0.2,0.3,0.2])

    stats = df_local.groupby('leaves')['len'].apply(calculate_custom_stats).apply(pd.Series)
    stats.columns = ['20th', '30th', 'Median', '70th', '80th']

    for i, (name, row) in enumerate(stats.iterrows(), start=0):
        fig.add_trace(go.Box(x=[name]*5, 
                             y=[row['20th'], row['30th'], row['Median'], row['70th'], row['80th']],
                             name="local",offsetgroup='A',
                             marker_color=bar_colors[0],
                             boxpoints=False), 
                      row=1, col=1)

    stats = df_boosted_local.groupby('leaves')['len'].apply(calculate_custom_stats).apply(pd.Series)
    stats.columns = ['20th', '30th', 'Median', '70th', '80th']

    for i, (name, row) in enumerate(stats.iterrows(), start=0):
        fig.add_trace(go.Box(x=[name]*5,  
                             y=[row['20th'], row['30th'], row['Median'], row['70th'], row['80th']],
                             name="localb",offsetgroup='B',
                             marker_color=bar_colors[1],
                             boxpoints=False),  
                      row=1, col=1)
    stats = df_local.groupby('leaves')['log_ratio'].apply(calculate_custom_stats).apply(pd.Series)
    stats.columns = ['20th', '30th', 'Median', '70th', '80th']

    for i, (name, row) in enumerate(stats.iterrows(), start=0):
        fig.add_trace(go.Box(x=[name]*5,  
                             y=[row['20th'], row['30th'], row['Median'], row['70th'], row['80th']],
                             name="log(local/localb)",offsetgroup='C',
                             marker_color=bar_colors[2],
                             boxpoints=False),  
                      row=1, col=2)

    stats = df_cqr.groupby('leaves')['len'].apply(calculate_custom_stats).apply(pd.Series)
    stats.columns = ['20th', '30th', 'Median', '70th', '80th']

    for i, (name, row) in enumerate(stats.iterrows(), start=0):
        fig.add_trace(go.Box(x=[name]*5,  
                             y=[row['20th'], row['30th'], row['Median'], row['70th'], row['80th']],
                             name="cqr",offsetgroup='A',
                             marker_color=bar_colors[0],
                             boxpoints=False), 
                      row=1, col=3)

    stats = df_boosted_cqr.groupby('leaves')['len'].apply(calculate_custom_stats).apply(pd.Series)
    stats.columns = ['20th', '30th', 'Median', '70th', '80th']

    for i, (name, row) in enumerate(stats.iterrows(), start=0):
        fig.add_trace(go.Box(x=[name]*5, 
                             y=[row['20th'], row['30th'], row['Median'], row['70th'], row['80th']],
                             name="cqrb",offsetgroup='B',
                             marker_color=bar_colors[1],
                             boxpoints=False),  
                      row=1, col=3)
        
    stats = df_cqr.groupby('leaves')['log_ratio'].apply(calculate_custom_stats).apply(pd.Series)
    stats.columns = ['20th', '30th', 'Median', '70th', '80th']

    for i, (name, row) in enumerate(stats.iterrows(), start=0):
        fig.add_trace(go.Box(x=[name]*5,  
                             y=[row['20th'], row['30th'], row['Median'], row['70th'], row['80th']],
                             name="log(cqr/cqrb)",offsetgroup='C',
                             marker_color=bar_colors[2],
                             boxpoints=False),  
                      row=1, col=4)
        
    # Update Layout 
    
    fig.update_layout(
        boxmode='group',
        height=350,width=1000,
        showlegend=False,
        margin=dict(t=100),
        plot_bgcolor='white', 
        paper_bgcolor='white'
    )
    for i in range(1, 5):  
        fig.update_xaxes(showgrid=True, gridcolor='lightgrey', linecolor='black', 
                         linewidth=1, mirror=True, tickmode='linear', dtick=1, 
                         row=1, col=i)
        fig.update_yaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, 
                         linecolor='black', linewidth=1, mirror=True, row=1, col=i)
        fig.update_yaxes(zeroline=True, zerolinecolor='lightgrey', zerolinewidth=0.1, row=1, col=i)
    

    fig.update_layout(boxgroupgap=0.25, boxgap=0.6,margin=margin)
    fig.add_annotation(
        text="Local Power",
        xref="x1",
        yref="paper",
        x=2.5,
        y=y_sub,
        font=dict(
            family="Arial, sans-serif",
            size=16,
            color="black"
        ),
        showarrow=False,
        align="center",
        xanchor="center",
        yanchor="bottom"
    )
    fig.add_shape(type="rect",
                  xref="paper", yref="paper",
                  x0=-0.0005, y0=1, x1=0.2555, y1=1.15, 
                  line=dict(color="black",width=1),
                  fillcolor="lightgray")

    fig.add_annotation(
        text="Local Log Ratio",  
        xref="x2",  
        yref="paper",  
        x=2.5,  
        y=y_sub,  
        font=dict(
            family="Arial, sans-serif",  
            size=16,  
            color="black"  
        ),
        showarrow=False, 
        align="center",  
        xanchor="center",  
        yanchor="bottom"  
    )
    fig.add_shape(type="rect",
                  xref="paper", yref="paper",
                  x0=0.3044, y0=1, x1=0.4755, y1=1.15, 
                  line=dict(color="black",width=1),
                  fillcolor="lightgray")

    fig.add_annotation(
        text="CQR Power", 
        xref="x3",  
        yref="paper", 
        x=2.5, 
        y=y_sub,  
        font=dict(
            family="Arial, sans-serif", 
            size=16, 
            color="black" 
        ),
        showarrow=False,  
        align="center", 
        xanchor="center", 
        yanchor="bottom" 
    )
    fig.add_shape(type="rect",
                  xref="paper", yref="paper",
                  x0=0.5244, y0=1, x1=0.7804, y1=1.15,  
                  line=dict(color="black",width=1),
                  fillcolor="lightgray")

    fig.add_annotation(
        text="CQR Log Ratio",  
        xref="x4",  
        yref="paper",  
        x=2.5,  
        y=y_sub, 
        font=dict(
            family="Arial, sans-serif", 
            size=16, 
            color="black" 
        ),
        showarrow=False, 
        align="center", 
        xanchor="center",  
        yanchor="bottom",  
    )
    fig.add_shape(type="rect",
                  xref="paper", yref="paper",
                  x0=0.8292, y0=1, x1=1.0003, y1=1.15,  
                  line=dict(color="black",width=1),
                  fillcolor="lightgray")
    
    bar_colors = ['#6baed6', '#fd8d3c', '#31a354']
    
    
    # Legend 
    
    labels = ['Classical', 'Boosted', 'Log Ratio']

    legend_x_starts = [0.32, 0.44, 0.55]  
    legend_y = 1.25  

    for i, color in enumerate(bar_colors):
        fig.add_shape(type="rect",
                      xref="paper", yref="paper",
                      x0=legend_x_starts[i], y0=legend_y,
                      x1=legend_x_starts[i] + 0.03, y1=legend_y + 0.05,  
                      line=dict(color=color, width=2),
                      fillcolor=color)

    for i, label in enumerate(labels):
        fig.add_annotation(xref="paper", yref="paper",
                           x=legend_x_starts[i] + 0.04, y=legend_y + 0.025, 
                           text=label,
                           font=dict(size=14),
                           showarrow=False,
                           xanchor="left",
                           yanchor="middle")
    fig.add_annotation(xref="paper", yref="paper",
                           x=0.4, y=-0.2,  
                           text='regression tree leaf',
                           font=dict(size=14),
                           showarrow=False,
                           xanchor="left",
                           yanchor="middle")
    if path:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directories: {directory}")
        if scale:
            fig.write_image(path,scale=scale)
        else:
            fig.write_image(path)

    fig.show()

    
def plot_len_data_process(data_name,ratio,seed,results_local,results_cqr,miscov_rate=0.1):
    ## load data
    params = dict()
    params['data']   = data_name                
    params['ratio']  = ratio                    
    params['seed']   = seed                  

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X_train, y_train, X_cal, y_cal, X_test, y_test= get_real_data(params)
    print("ratio: ",ratio)
    
    ## run Regression tree
    X_traincal=np.concatenate((X_train, X_cal), axis=0)
    y_traincal=np.concatenate((y_train, y_cal), axis=0)
    regressor = DecisionTreeRegressor(max_leaf_nodes=4,min_samples_leaf=int(X_traincal.shape[0]/10))
    regressor.fit(X_traincal, y_traincal)
    leaves = regressor.apply(X_test)  
    leaf_data_points = {}
    for leaf_index in set(leaves):
        leaf_data_points[leaf_index] = X_test[leaves == leaf_index, :]
    data_points_in_first_leaf = leaf_data_points[list(set(leaves))[0]]

    unique_leaves = np.unique(leaves)
    class_mapping = {key: val for val, key in enumerate(unique_leaves, start=1)}
    mapped_leaves = np.vectorize(class_mapping.get)(leaves)
    
    ## process boosted resuls 
    df_local = {k: results_local[k] for k in results_local if k in ('local_upper', 'local_lower')}
    df_local=pd.DataFrame(df_local)
    df_local['leaves']=mapped_leaves
    df_local['ifboosted']='local'
    df_local['len']=df_local['local_upper']-df_local['local_lower']
    print("local_len: ",results_local['local_len'])
    print("local_cov: ",results_local['local_cov'])
    df_boosted_local = {k: results_local[k] for k in results_local if k in ('boosted_upper', 'boosted_lower')}
    df_boosted_local=pd.DataFrame(df_boosted_local)
    df_boosted_local['leaves']=mapped_leaves
    df_boosted_local['ifboosted']='localb'
    df_boosted_local['len']=df_boosted_local['boosted_upper']-df_boosted_local['boosted_lower']
    print("boosted_len: ",results_local['boosted_len'])
    print("boosted_cov: ",results_local['boosted_cov'])
    ave_imp = np.mean(df_local['len']-df_boosted_local['len'])/np.mean(df_local['len'])
    
    df_cqr = {k: results_cqr[k] for k in results_cqr if k in ('cqr_upper', 'cqr_lower')}
    df_cqr=pd.DataFrame(df_cqr)
    df_cqr['leaves']=mapped_leaves
    df_cqr['ifboosted']='cqr'
    df_cqr['len']=df_cqr['cqr_upper']-df_cqr['cqr_lower']
    print("cqr_len: ",results_cqr['cqr_len'])
    print("cqr_cov: ",results_cqr['cqr_cov'])
    df_boosted_cqr = {k: results_cqr[k] for k in results_cqr if k in ('boosted_upper', 'boosted_lower')}
    df_boosted_cqr=pd.DataFrame(df_boosted_cqr)
    df_boosted_cqr['leaves']=mapped_leaves
    df_boosted_cqr['ifboosted']='cqrb'
    df_boosted_cqr['len']=df_boosted_cqr['boosted_upper']-df_boosted_cqr['boosted_lower']
    print("boosted_len: ",results_cqr['boosted_len'])
    print("boosted_cov: ",results_cqr['boosted_cov'])
    ave_imp_cqr = np.mean(df_cqr['len']-df_boosted_cqr['len'])/np.mean(df_cqr['len'])

    print("Average Improvement on Local (%): ", round(ave_imp*100,2))
    print("Average Improvement on CQR (%): ", round(ave_imp_cqr*100,2))
    
    df=pd.concat([df_local, df_boosted_local], ignore_index=True)
    log_ratio=np.log(df_local['len']/ df_boosted_local['len'])
    df_local['log_ratio']=log_ratio

    df=pd.concat([df_cqr, df_boosted_cqr], ignore_index=True)
    log_ratio_cqr=np.log(df_cqr['len']/ df_boosted_cqr['len'])
    df_cqr['log_ratio']=log_ratio_cqr
    return df_local,df_boosted_local,df_cqr,df_boosted_cqr       

def calculate_custom_stats(group):
    thirtieth = group.quantile(0.3)
    median = group.median()
    seventieth = group.quantile(0.7)
    twentieth_percentile = group.quantile(0.2)
    eighttieth_percentile = group.quantile(0.8)
    return twentieth_percentile,thirtieth, median, seventieth, eighttieth_percentile

