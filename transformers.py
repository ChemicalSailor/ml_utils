# -*- coding: utf-8 -*-
"""Additional Scikit-Learn transformers."""

import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import FunctionTransformer as FunctionTransformer_Base
from .named_features import with_feature_names


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


class ColumnSelector( BaseEstimator, TransformerMixin ):
    """
    Transform a dataframe by including or excluding columns.

    Use this transformer to select which columns of a dataframe to keep.
    This is primarily controlled by providing the column names as the keys of a dict
    and the values as a boolean indication whether to include the column.

    Each column specified in `column_map` is made available as a parameter using the
    double underscore syntax allowing each column to be set atomically using `set_params`
    and thus searched during hyperparameter tuning. Columns not specified in `column_map`
    cannot be accessed in this way.

    Parameters
    ----------
    column_map : dict
        A dict of column names as keys and boolean values. A value of `True` passes
        through the column unchanged; values of `False` will drop the column.
    remainder : {‘drop’, ‘passthrough’}, default=’drop’
        The action to perform on columns not specified in `column_map`.

    Attributes
    ----------
    column_mapping_ : dict
        The column map after fitting. This will include all columns present on `X`.

    Examples
    --------
    Create a dataframe and specify a column map to include or drop columns:

    >>> X = pd.DataFrame( np.random.normal( size=(50,5) ), columns=list('abcde') )
    >>> trf = ColumnSelector( { 'a': True, 'b': False }, remainder='passthrough' )
    >>> trf.fit_transform( X ).columns
    Index(['a', 'c', 'd', 'e'], dtype='object')

    Set column inclusion on an individual basis. This only works for columns specified in
    `column_map`:

    >>> trf.set_params( column_map__a=False )
    >>> trf.fit_transform( X ).columns
    Index(['c', 'd', 'e'], dtype='object')

    """

    def __init__( self, column_map, *, remainder='drop' ):
        # self.transformers = transformers
        self.column_map = column_map
        self.remainder = remainder

    def _validate_params( self, X, y=None ):
        is_dataframe = isinstance( X, pd.DataFrame )
        if not is_dataframe:
            raise TypeError( "Parameter 'X' must be of type 'DataFrame'" )

        for name in self.column_map.keys():
            if name not in X.columns:
                raise ValueError( f"'{name}' is not present in the dataframe columns" )

        if self.remainder not in ('drop', 'passthrough'):
            raise ValueError(
                f"Hyperparameter 'remainder' must be 'drop' or 'passthrough'; got '{self.remainder}'"
            )

    def fit( self, X, y=None ):
        self._validate_params( X, y )

        if self.remainder == 'drop':
            self.column_mapping_ = { name: False for name in X.columns }
        else:
            self.column_mapping_ = { name: True for name in X.columns }

        self.column_mapping_.update( self.column_map )

        return self

    def get_feature_names( self ):
        check_is_fitted( self, 'column_mapping_' )
        return tuple( self.column_mapping_.keys() )

    def get_params( self, deep=True ):
        params = super().get_params( deep )
        if deep:
            mutations = { f'column_map__{k}': v for k, v in self.column_map.items() }
            params.update( mutations )
        return params

    def set_params( self, **params ):
        column_mutations = {}
        for key in tuple( params.keys() ):
            match = re.search( '(?<=column_map__)\w+', key )
            if match:
                column_mutations[ match[0] ] = params.pop( key )
        # apply the non-mutation params first
        super().set_params( **params )
        # then apply the mutations
        for k, v in column_mutations.items():
            # try getting the key first to check it exists
            self.column_map[k]
            self.column_map[k] = v
        return self

    def transform( self, X ):
        check_is_fitted( self, 'column_mapping_' )
        column_mask = list( self.column_mapping_.values() )
        return X.loc[:, column_mask]

__all__ = [ HyperparameterMixin, DumbTransformer, FunctionTransformer, ColumnSelector ]