import sys
import numpy as np
import pdb

# CQR error function
class QR_errfun():
    """Calculates conformalized quantile regression error.

    Conformity scores:

    .. math::
        max{\hat{q}_low - y, y - \hat{q}_high}

    """
    def __init__(self):
        super(QR_errfun, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:,0]
        y_upper = prediction[:,-1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high,error_low)
        return err

    def apply_inverse(self, nc, significance):
        q = np.quantile(nc, np.minimum(1.0, (1.0-significance)*(nc.shape[0]+1.0)/nc.shape[0]))
        return np.vstack([q, q])

# CQR-m error function
class QRm_errfun():
    """Calculates conformalized quantile regression error with median.

    Conformity scores:

    .. math::
        max{(\hat{q}_low-y) / (\hat{q}_median-\hat{q}_low), (y-\hat{q}_high) / (\hat{q}_high-\hat{q}_median)}

    """
    def __init__(self):
        super(QRm_errfun, self).__init__()

    def apply(self, prediction, y):
        eps = 1e-6
        y_lower = prediction[:,0]
        y_median = prediction[:,1]
        y_upper = prediction[:,-1]
        error_low = (y_lower - y) / (y_median - y_lower + eps)
        error_high = (y - y_upper) / (y_upper - y_median + eps)
        err = np.maximum(error_high, error_low)
        return err

    def apply_inverse(self, nc, significance):
        q = np.quantile(nc, np.minimum(1.0, (1.0-significance)*(nc.shape[0]+1.0)/nc.shape[0]))
        return np.vstack([q, q])

# CQR-r error function
class QRr_errfun():
    """Calculates rescaled conformalized quantile regression error.

    Conformity scores:

    .. math::
        max{(\hat{q}_low-y) / (\hat{q}_high-\hat{q}_low), (y-\hat{q}_high) / (\hat{q}_high-\hat{q}_low)}

    """
    def __init__(self):
        super(QRr_errfun, self).__init__()

    def apply(self, prediction, y):
        eps = 1e-6
        y_lower = prediction[:,0]
        y_upper = prediction[:,-1]
        scaling_factor = y_upper - y_lower + eps
        error_low = (y_lower - y) / scaling_factor
        error_high = (y - y_upper) / scaling_factor
        err = np.maximum(error_high, error_low)
        return err

    def apply_inverse(self, nc, significance):
        q = np.quantile(nc, np.minimum(1.0, (1.0-significance)*(nc.shape[0]+1.0)/nc.shape[0]))
        return np.vstack([q, q])

class ConformalizedQR:
    def __init__(self, model, model_tuning, X, y, idx_train, idx_cal, significance, verbose=False):
        self.model = model
        self.X = X
        self.y = y
        self.idx_train = idx_train
        self.idx_cal = idx_cal
        self.verbose = verbose

        # Fit model using training set
        print("Fitting QR model with " + str(len(self.idx_train)) + " observations...")
        sys.stdout.flush()
        self.model.fit(self.X[self.idx_train,:], self.y[self.idx_train])
        print("Fitting complete.")
        sys.stdout.flush()

        # Make predictions on training dataset
        self.predictions_train =  self.model.predict(self.X[self.idx_train,:])
        
        # Make predictions on calibration dataset
        print("Making predictions using calibration data... ", end="")
        sys.stdout.flush()
        self.predictions_cal = self.model.predict(self.X[self.idx_cal,:])
        print("done.")
        sys.stdout.flush()

    def predict(self, X_test, y_test, significance, method = "CQR"):
        # Define scoring function
        if method == "CQR":
            scorer = QR_errfun()
        elif method == "CQRm":
            scorer = QRm_errfun()
        elif method == "CQRr":
            scorer = QRr_errfun()
        else:
            print("Uknown method.")
            sys.exit()

        print("Computing score correction: ", end="")
        sys.stdout.flush()
        # Compute conformity scores on calibration dataset
        scores = scorer.apply(self.predictions_cal, self.y[self.idx_cal])
        # Compute correction factor based on scores
        score_correction = scorer.apply_inverse(scores, significance)
        print("%.3f" % (score_correction[0]))
        sys.stdout.flush()

        # Compute QR prediction intervals on test data
        print("Making predictions using test data... ", end="")
        sys.stdout.flush()
        predictions_test = self.model.predict(X_test)
        print("done.")
        sys.stdout.flush()

        # Conformalize prediction intervals on test data
        if method == "CQR":
            lower = predictions_test[:,0] - score_correction[0,0]
            upper = predictions_test[:,-1] + score_correction[1,0]
        elif method == "CQRm":
            eps = 1e-6
            pred = predictions_test
            lower = pred[:,0] - score_correction[0,0] * (pred[:,1] - pred[:,0] + eps)
            upper = pred[:,-1] + score_correction[1,0] * (pred[:,-1] - pred[:,1] + eps)
        elif method == "CQRr":
            eps = 1e-6
            pred = predictions_test
            scaling_factor = pred[:,-1] - pred[:,0] + eps
            lower = pred[:,0] - score_correction[0,0] * scaling_factor
            upper = pred[:,-1] + score_correction[1,0] * scaling_factor
        else:
            print("Uknown method.")
            sys.exit()

        return lower, upper

    ### added for BoostedCP
    def predicted_quantiles(self, X_test, y_test, significance):        
        # Define scoring function
        scorer = QR_errfun()
        print("Making predictions using test data... ", end="")
        sys.stdout.flush()
        predictions_test = self.model.predict(X_test)
        print("done.")
        sys.stdout.flush()
        return self.predictions_train, self.predictions_cal, predictions_test

   