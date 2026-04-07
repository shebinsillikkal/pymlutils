"""
PyMLUtils — Plotting Utilities
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List

def plot_feature_importance(model, feature_names: List[str], top_n: int = 20,
                             title: str = "Feature Importance", figsize=(10, 6)):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, labels: Optional[List] = None,
                           title: str = "Confusion Matrix", figsize=(8, 6)):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_proba, title: str = "ROC Curve", figsize=(8, 6)):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig
