import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from .utils import default_none_kwargs


class LinearTrendTransformer(TransformerMixin, BaseEstimator):
    """Transform features by removing linear trends.

    Uses Ordinary least squares Linear Regression as implemented in
    sklear.linear_model.LinearRegression.

    Parameters
    ----------
    **lr_kwargs
        Keyword arguments to pass to sklearn.linear_model.LinearRegression

    Attributes
    ----------
    lr_model_ : sklearn.linear_model.LinearRegression
        Linear Regression object.
    """

    def __init__(self, lr_kwargs=None):
        self.lr_kwargs = lr_kwargs

    def fit(self, X, y=None):
        """Compute the linear trend.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            Training data.
        """
        X = self._validate_data(X)
        kwargs = default_none_kwargs(self.lr_kwargs)
        self.lr_model_ = LinearRegression(**kwargs)
        self.lr_model_.fit(np.arange(len(X)).reshape(-1, 1), X)
        return self

    def transform(self, X):
        """Perform transformation by removing the trend.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            The data that should be detrended.
        """
        # validate input data
        check_is_fitted(self)
        X = self._validate_data(X)
        return X - self.trendline(X)

    def inverse_transform(self, X):
        """Add the trend back to the data.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            The data that should be transformed back.
        """
        # validate input data
        check_is_fitted(self)
        X = self._validate_data(X)
        return X + self.trendline(X)

    def trendline(self, X):
        """helper function to calculate a linear trendline"""
        X = self._validate_data(X)
        return self.lr_model_.predict(np.arange(len(X)).reshape(-1, 1))

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_methods_subset_invariance': 'because',
                'check_methods_sample_order_invariance': 'temporal order matters',
            }
        }

class RunningTrendTransformer(TransformerMixin, BaseEstimator):
    """Transform features by removing running linear trends.

    Parameters
    ----------
    window : int
        Window size for running trend calculation.
    """

    def __init__(self, window=9):
        self.window = window
        self.half_window = int(window / 2)

    def fit(self, X, y=None):
        """Compute the linear trend.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            Training data.
        """
        # x.rolling(9, center=True, min_periods=1).mean()
        X = self._validate_data(X)
        X_mean = np.mean(X[:,0])
        X_null = np.repeat(X_mean, self.half_window)[:, np.newaxis]
        X_expand = np.concatenate((X_null, X, X_null), axis=0)
        self.running_trendline = (np.convolve(X_expand[:,0], np.ones(self.window), 'valid') / self.window)[:, np.newaxis]
        return self

    def transform(self, X):
        """Perform transformation by removing the trend.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            The data that should be detrended.
        """
        # validate input data
        check_is_fitted(self)
        X = self._validate_data(X)
        assert len(X) == len(self.running_trendline)
        return X - self.running_trendline

    def inverse_transform(self, X):
        """Add the trend back to the data.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            The data that should be transformed back.
        """
        # validate input data
        check_is_fitted(self)
        X = self._validate_data(X)
        assert len(X) == len(self.running_trendline)
        return X + self.running_trendline