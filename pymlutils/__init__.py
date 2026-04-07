"""
PyMLUtils — Open Source ML Utilities Library
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
from .preprocessing import SmartScaler, FeatureSelector, OutlierRemover
from .evaluation import ModelEvaluator, CrossValidator
from .plotting import plot_feature_importance, plot_confusion_matrix, plot_roc_curve

__version__ = "1.2.0"
__author__ = "Shebin S Illikkal"
__all__ = [
    "SmartScaler", "FeatureSelector", "OutlierRemover",
    "ModelEvaluator", "CrossValidator",
    "plot_feature_importance", "plot_confusion_matrix", "plot_roc_curve"
]
