"""
PyMLUtils — Classification Report Helper
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from typing import Optional


class ClassificationReport:
    def __init__(self, model, X_test, y_test, threshold: float = 0.5):
        self.model = model
        self.y_true = y_test
        self.y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int) \
                      if hasattr(model, 'predict_proba') else model.predict(X_test)
        self.y_proba = model.predict_proba(X_test)[:, 1] \
                       if hasattr(model, 'predict_proba') else None

    def show(self) -> str:
        report = classification_report(self.y_true, self.y_pred)
        if self.y_proba is not None:
            auc = roc_auc_score(self.y_true, self.y_proba)
            print(f"ROC-AUC: {auc:.4f}\n")
        print(report)
        return report

    def plot_roc(self, figsize=(7, 5)):
        if self.y_proba is None:
            print("Model does not support predict_proba")
            return
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        auc = roc_auc_score(self.y_true, self.y_proba)
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0,1],[0,1],'k--', lw=1)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve'); plt.legend(); plt.tight_layout(); plt.show()

    def plot_confusion(self, figsize=(6, 5)):
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=figsize)
        disp.plot(ax=ax, colorbar=False)
        plt.title('Confusion Matrix'); plt.tight_layout(); plt.show()
