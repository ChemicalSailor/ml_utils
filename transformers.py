# -*- coding: utf-8 -*-
"""Additional Scikit-Learn transformers."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer as FunctionTransformer_Base
from named_features import with_feature_names


class HyperparameterMixin:
    """
    Mixin class to ease setting model attributes in a constructor.

    This mixin provides one method `_set_args( args: dict )` which should be used
    in conjunction with `locals()` to set attributes in `__init__`.

    Example
    -------
    ```
    MyEstimator(BaseEstimator, HyperparameterMixin):
         def __init__(self, arg1=None, arg2=None):
             self._set_args( locals() )
    ```
    """

    def _set_args( self, args: dict ):
        """
        Set attributes (hyperparameters) from a dictionary.

        Parameters
        ----------
        args : dict
            Arguments and values to set as attributes.

        """
        try:
            args.pop( 'self' )
        except KeyError:
            pass
        for arg, val in args.items():
            setattr( self, arg, val )


class DumbTransformer( BaseEstimator, TransformerMixin ):
    """
    Base transformer class.

    Implements generic `fit(X, y)` and `transform(X)` methods that don't do anything, in
    addition to `fit_transform(X, y)` from `TransformerMixin`.

    """

    def fit( self, X, y=None ):
        return self

    def transform( self, X ):
        return X


class FunctionTransformer( with_feature_names( FunctionTransformer_Base ) ):
    """
    An extension of Scikit-Learn's FunctionTransformer with support for feature names.

    This implementation supports an additional keyword argument `feature_names_func`
    in the constructor to specify a function that transforms the feature names.
    Feature names are inferred (from DataFrame column names) or generated (for a
    NumPy array)when `fit()` is called. The transformed feature names can be
    accessed using the `get_feature_names()` method.

    Parameters
    ----------
    feature_names_func : callable
        Function that transforms feature names. If not provided feature names are
        passed through unchanged.

    Attributes
    ----------
    feature_names_ : list
        Inferred or generated feature names.

    """

    def __init__( self, func=None, inverse_func=None, *, feature_names_func=None, validate=False,
                  accept_sparse=False, check_inverse=True, kw_args=None,
                  inv_kw_args=None ):
        args = {
            'func': func,
            'inverse_func': inverse_func,
            'validate': validate,
            'accept_sparse': accept_sparse,
            'check_inverse': check_inverse,
            'kw_args': kw_args,
            'inv_kw_args': inv_kw_args,
        }
        super().__init__( **args )
        self.feature_names_func = feature_names_func

    def get_feature_names( self ):
        """Return transformed feature names.

        Returns
        -------
        List[str]
            The result of applying `feature_names_func` to feature names.

        """
        if self.feature_names_func:
            return list( map( self.feature_names_func, self.feature_names_ ) )
        return self.feature_names_
