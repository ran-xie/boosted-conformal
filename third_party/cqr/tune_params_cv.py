import sys,pdb
import numpy as np
from cqr import helper
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def CV_quntiles_rf(params, X, y, target_coverage, grid_q, test_ratio, random_state, coverage_factor=1.0):
    """ Tune the low and high quantile level parameters of quantile random
        forests method, using cross-validation
    
    Parameters
    ----------
    params : dictionary of parameters
            params["random_state"] : integer, seed for splitting the data 
                                     in cross-validation. Also used as the
                                     seed in quantile random forest (QRF)
            params["min_samples_leaf"] : integer, parameter of QRF
            params["n_estimators"] : integer, parameter of QRF
            params["max_features"] : integer, parameter of QRF
    X : numpy array, containing the training features (nXp)
    y : numpy array, containing the training labels (n)
    target_coverage : desired coverage of prediction band. The output coverage
                      may be smaller if coverage_factor <= 1, in this case the
                      target will be modified to target_coverage*coverage_factor
    grid_q : numpy array, of low and high quantile levels to test
    test_ratio : float, test size of the held-out data
    random_state : integer, seed for splitting the data in cross-validation.
                   Also used as the seed in QRF.
    coverage_factor : float, when tuning the two QRF quantile levels one may
                      ask for prediction band with smaller average coverage,
                      equal to coverage_factor*(q_high - q_low) to avoid too
                      conservative estimation of the prediction band
    
    Returns
    -------
    best_q : numpy array of low and high quantile levels (length 2)
    
    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.
    
    """
    target_coverage = coverage_factor*target_coverage

    rf = RandomForestQuantileRegressor(random_state=params["random_state"],
                                       min_samples_leaf=params["min_samples_leaf"],
                                       n_estimators=params["n_estimators"],
                                       max_features=params["max_features"])

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    n_folds = 10
    kf = KFold(n_splits=n_folds)
    folds = kf.split(X,y)

    coverage_values = np.zeros((len(grid_q), n_folds))
    length_values = np.zeros((len(grid_q), n_folds))

    fold_idx = 0
    for fold in folds:
        print("[CV DEBUG] fold " + str(fold_idx+1) + " of " + str(n_folds) + "... ", end="")
        sys.stdout.flush()

        idx_train = fold[0]
        idx_test = fold[1]
        X_train = X[idx_train,:]
        y_train = y[idx_train]
        X_test = X[idx_test,:]
        y_test = y[idx_test]

        rf.fit(X_train, y_train)

        for q_idx in range(len(grid_q)):
            q = grid_q[q_idx]
            y_lower = rf.predict(X_test, quantile=q[0])
            y_upper = rf.predict(X_test, quantile=q[-1])
            coverage, avg_length = helper.compute_coverage_len(y_test, y_lower, y_upper)
            coverage_values[q_idx,fold_idx] = coverage
            length_values[q_idx,fold_idx] = avg_length

        fold_idx = fold_idx+1

        print("done.")
        sys.stdout.flush()


    avg_coverage = coverage_values.mean(1)
    avg_length = length_values.mean(1)

    idx_under = np.where(avg_coverage<=target_coverage)[0]
    if len(idx_under)>0:
        best_idx = np.max(idx_under)
    else:
        best_idx = 0
    best_q = grid_q[best_idx]
    best_coverage = avg_coverage[best_idx]
    best_length = avg_length[best_idx]

    print("[CV DEBUG] best q " + str(best_q) + ", coverage " + str(best_coverage) + 
          ", length " + str(best_length))

    return best_q, best_coverage, best_length
