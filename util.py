# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:30:56 2020

@author: tfletcher
"""

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import itertools
from typing import Iterable, List

def iterative_column_selector( col_names: Iterable[str] ) -> List[List[str]]:
    n = len( col_names )
    it = itertools.chain.from_iterable( [ itertools.combinations( col_names, i ) for i in range(1, n+1) ] )
    return [ list( v ) for v in it ]

def if_column_exists( column_names ):
    """
    Helper for `ColumnTransformer` to only specify columns that exist in a dataframe `X`.

    Parameters
    ----------
    column_names : list
        The dataframe columns to select.

    Returns
    -------
    callable
        A function that returns a filtered list of column names given `X`.

    """
    def selector( X ):
        cols = [ name for name in X.columns if name in column_names ]
        return cols
    return selector