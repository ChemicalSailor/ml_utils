# -*- coding: utf-8 -*-
"""Functions to add support for feature names to Scikit-Learn estimators."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import List


def names_from_dataframe( X: pd.DataFrame ) -> List[str]:
    """Get feature names from the columns of a DataFrame.

    Parameters
    ----------
    X : DataFrame
        The training data.

    Returns
    -------
    List[str]
        The column names.

    """
    return list( X.columns )


def generate_names( X: np.ndarray ) -> List[str]:
    """
    Generate feature names for given training data.

    Feature names are given as `[x0, x1, ... xn]` matching the number of columns
    in `X`

    Parameters
    ----------
    X : array
        The training data.

    Returns
    -------
    List[str]
        Generated feature names.

    """
    n_samples, n_features = X.shape()
    names = [ 'x'+str(x) for x in range(1, n_features+1) ]
    return names


class FeatureNamesMixin:
    """Mixin class to add feature names support to an estimator."""

    def fit( self, X, y=None ):
        super().fit( X, y )
        if type( X ) == pd.DataFrame:
            self.feature_names_ = names_from_dataframe( X )
        else:
            self.feature_names_ = generate_names( X )
        return self

    def get_feature_names( self ) -> List[str]:
        """Return the feature names as a list."""
        return self.feature_names_


def with_feature_names( estimator=None, *, names=None ):
    """
    Decorate an estimator with support for named features.

    During training feature names are inferred from the training data when
    presented as a pandas dataframe, or generated when presented as a numpy
    array. Alternatively feature names can be provided to the decorator.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to subclass. When used as a decorator this parameter must
        remain as `None`.
    names : list[str], optional
        A list of custom feature names. Must be passed as a keyword argument.

    Returns
    -------
    Mixed
        A subclassed estimator, or a function to subclass an estimator.

    Examples
    --------
    As a decorator:

    ```
    @with_feature_names
    class MyEstimator( BaseEstimator ):
        pass
    ```

    With custom names:

    ```
    feature_names = ['a', 'b', 'c']
    # names must be passed as keyword argument
    @with_feature_names( names=feature_names )
    class MyEstimator( BaseEstimator ):
        pass
    ```

    May also be used as a function which is useful to decorate imported
    estimators:

    ```
    >>> with_feature_names( BaseEstimator )
    >>> with_feature_names( BaseEstimator, names=feature_names )
    >>> with_feature_names( names=feature_names )( BaseEstimator )
    ```

    """

    def _decorator( cls: BaseEstimator ):
        # TODO: check cls is actually an estimator
        class _WithFeatureNames( cls ):

            def fit( self, X, y=None ):
                super().fit( X, y )
                if names:
                    n_features = X.shape[1]
                    if n_features != len( names ):
                        raise ValueError( f'The length of the given feature names ({len( names)}) does not match the number of received features ({n_features})' )
                    self.feature_names_ = names
                elif type( X ) == pd.DataFrame:
                    self.feature_names_ = names_from_dataframe( X )
                else:
                    self.feature_names_ = generate_names( X )
                return self

            def get_feature_names( self ):
                return self.feature_names_

        _WithFeatureNames.__name__ = cls.__name__
        _WithFeatureNames.__qualname__ = cls.__qualname__
        _WithFeatureNames.__doc__ = cls.__doc__
        _WithFeatureNames.__module__ = cls.__module__
        return _WithFeatureNames

    _decorator.__name__ = with_feature_names.__name__
    _decorator.__qualname__ = with_feature_names.__qualname__
    _decorator.__doc__ = with_feature_names.__doc__
    _decorator.__module__ = with_feature_names.__module__
    if estimator is None:
        return _decorator
    return _decorator( estimator )
