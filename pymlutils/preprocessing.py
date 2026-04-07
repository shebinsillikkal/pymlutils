"""
PyMLUtils — Preprocessing Utilities
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Literal, Optional, List

class SmartScaler(BaseEstimator, TransformerMixin):
    """Automatically chooses the right scaler based on data distribution."""

    def __init__(self, method: Literal['auto', 'standard', 'minmax', 'robust'] = 'auto'):
        self.method = method
        self._scaler = None
        self._chosen_method = None

    def _detect_method(self, X: np.ndarray) -> str:
        from scipy import stats
        _, p = stats.normaltest(X.ravel())
        if p < 0.05:  # Not normal
            iqr = np.percentile(X, 75) - np.percentile(X, 25)
            if iqr > 0 and (X.max() - X.min()) / iqr > 10:
                return 'robust'
            return 'minmax'
        return 'standard'

    def fit(self, X, y=None):
        X = np.array(X)
        method = self.method if self.method != 'auto' else self._detect_method(X)
        self._chosen_method = method
        scalers = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
        self._scaler = scalers[method]()
        self._scaler.fit(X)
        return self

    def transform(self, X, y=None):
        return self._scaler.transform(X)

    @property
    def chosen_method(self):
        return self._chosen_method


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select features by importance threshold or top-K."""

    def __init__(self, method: Literal['variance', 'correlation', 'importance'] = 'variance',
                 threshold: float = 0.01, top_k: Optional[int] = None):
        self.method = method
        self.threshold = threshold
        self.top_k = top_k
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if self.method == 'variance':
            variances = X.var()
            mask = variances > self.threshold
        elif self.method == 'correlation' and y is not None:
            corrs = X.corrwith(pd.Series(y)).abs()
            mask = corrs > self.threshold
        else:
            mask = pd.Series([True] * X.shape[1], index=X.columns)

        self.selected_features_ = X.columns[mask].tolist()
        if self.top_k:
            self.selected_features_ = self.selected_features_[:self.top_k]
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return X[self.selected_features_]


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove or cap outliers using IQR or Z-score method."""

    def __init__(self, method: Literal['iqr', 'zscore'] = 'iqr',
                 action: Literal['remove', 'cap'] = 'cap', factor: float = 1.5):
        self.method = method
        self.action = action
        self.factor = factor
        self._bounds = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in X.select_dtypes(include=np.number).columns:
            if self.method == 'iqr':
                q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                iqr = q3 - q1
                self._bounds[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
            else:
                mean, std = X[col].mean(), X[col].std()
                self._bounds[col] = (mean - self.factor * std, mean + self.factor * std)
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        for col, (lower, upper) in self._bounds.items():
            if self.action == 'cap':
                X[col] = X[col].clip(lower, upper)
            else:
                X = X[(X[col] >= lower) & (X[col] <= upper)]
        return X
