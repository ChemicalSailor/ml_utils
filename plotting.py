# -*- coding: utf-8 -*-
"""Matplotlib plots for model analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score


def binary_confusion_matrix( y_true, y_predicted ):
    """
    Computes the confusion matrix for a binary classification system.
    Returns as a pandas dataframe for pretty printing with labels.
    """
    m = confusion_matrix( y_true, y_predicted )
    labels = ('Negative', 'Positive')
    i = pd.MultiIndex.from_product( ( ('Actual',), labels ) )
    c = pd.MultiIndex.from_product( ( ('Predicted',), labels ) )
    df = pd.DataFrame( m, index=i, columns=c )
    return df


def plot_confusion_matrix( y_true, y_predicted ):

    m = confusion_matrix( y_true, y_predicted )
    norm = m / m.sum( axis=1, keepdims=True )
    errors = norm.copy()
    np.fill_diagonal( errors, np.nan )

    fig, ax = plt.subplots( 1, 3, figsize=(10, 10) )
    for axx, mx in zip( ax, [m, norm, errors] ):
        axx.matshow( mx )
        axx.set_ylabel( 'Actual' )
        axx.set_xlabel( 'Predicted' )
    ax[0].set_title( 'Confusion Matrix' )
    ax[1].set_title( 'Normalised' )
    ax[2].set_title( 'Normalised, Errors Only' )
    plt.tight_layout()
    plt.show()
    return m


def plot_precision_recall_score( y, y_scores ):
    """
    Plots precison and recall against the varying score/threshold.
    """
    precision, recall, score = precision_recall_curve( y, y_scores )
    # precision and recall have 0 and 1 added to the end, so they get trimmed off
    plt.plot( score, precision[:-1], 'b--', label='Precesion' )
    plt.plot( score, recall[:-1], 'g-', label='Recall' )
    plt.xlabel( 'Score' )
    plt.legend()
    plt.ylim((0, 1))
    plt.show()


def plot_precision_recall( y, y_scores ):
    """
    Plots precision against recall.

    Most useful when positive class is rare or number of false positives is
    more important than false negatives.
    """
    precision, recall, score = precision_recall_curve( y, y_scores )
    plt.plot( recall[:-1], precision[:-1] )
    plt.axis( [0, 1, 0, 1 ] )
    plt.xlabel( 'Recall' )
    plt.ylabel( 'Precision' )
    plt.show()


def plot_ROC( y, y_scores ):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Most useful when number of positives ~= number of negatives.
    A dumb classifier is represented by the diagonal line.
    """
    fpr, tpr, scores = roc_curve( y, y_scores )
    auc = roc_auc_score( y, y_scores )
    plt.plot( fpr, tpr )
    plt.plot( [0, 1], [0, 1], 'k--' )
    plt.axis( [0, 1, 0, 1 ] )
    plt.ylabel( 'Recall (TPR)' )
    plt.xlabel( '1-Specifity (FPR)' )
    plt.text( 0.6, 0.1, 'AUC = {:.3f}'.format( auc ) )
    plt.show()
