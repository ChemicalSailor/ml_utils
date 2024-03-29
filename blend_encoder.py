"""
Modified one-hot encoder for blends.

@author: T Fletcher
"""

# TODO: Test missing sample
#   check_array checks for missing values (np.nan or pd.NA) provided input is 'object' dtype
#   need to decide if input should be cast, in which case there should be string replacement for empty values
# TODO: handle unknown solvents passed to transform
#   currently implicitedly converted to 'none' and dropped meaning validation fails

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted, check_array, _check_feature_names_in
import re
from typing import List, Tuple, Iterable, Optional, Union
from numpy.typing import ArrayLike, NDArray


class _BlendParser:

    def __init__(
            self,
            delimit: str = ':',
            case_sensitive: bool = False,
            parse_quantities: bool = True
            ):
        self.delimit = delimit
        self.case_sensitive = case_sensitive
        self.parse_quantities = parse_quantities

    def _get_regex( self ):
        """Compile regex for given state."""
        flag = re.IGNORECASE if not self.case_sensitive else 0
        regex_str = f'^([\w,\- {self.delimit}()\[\]]*?)[(\[ ]+([\d{self.delimit}.]*)[)\]]?$'
        regex = re.compile( regex_str, flag )
        return regex

    def __call__( self, value: str ) -> Union[ Tuple[ List[str], List[float] ], List[str] ]:
        """
        Parse a string with blend information.

        Parameters
        ----------
        value : str
            String to parse.

        Returns
        -------
        Tuple[ List[str], List[float] ]
            A list of found components and a list of matching quantities.
        List[str]
            A list of found components when `parse_quantities=False`.

        Raises
        ------
        ValueError
            When the given string does not match the expected components
            quantities pattern.
        ValueError
            When the length of found components and found quantities do not
            match.

        """
        c, q = self._parse( self._get_regex(), value )
        if self.parse_quantities:
            return c, q
        return c

    def _parse( self, regex, value: str ) -> Tuple[ List[str], Optional[List[float]] ]:
        """Parse string with pre-compiled regex."""
        # handle the pure solvent case first
        if self.delimit not in value:
            # assume that the whole string is the solvent name
            if self.parse_quantities:
                return [value], [1.0]
            return [value], None

        # try to match a complete string and fall back to the partial string
        # if not matching quantities
        try:
            components, quantities = regex.match( value ).groups()
        except AttributeError:
            if self.parse_quantities:
                raise ValueError(
                    "Could not match components and quantities pattern in the "
                    "string `{}`; validate string format or use "
                    "`parse_quantities=False`". format( value )
                    )
            components = value

        components = components.split( self.delimit )

        if not self.parse_quantities:
            return components, None

        quantities = quantities.split( self.delimit )

        n_solv = len(components)
        n_quan = len(quantities)
        if n_solv != n_quan:
            raise ValueError(
                "Mismatch in solvent blend string '{}', found {} components "
                "and {} quantities".format( value, n_solv, n_quan)
            )

        return components, [ float(x) for x in quantities ]

    def map( self, x: ArrayLike ) -> Tuple[ NDArray, NDArray ]:
        x = np.array( x )
        if x.ndim != 1:
            raise ValueError( f"Expected 1D array, got {x.ndim}D" )

        regex = self._get_regex()

        max_components = 0
        components = []
        quantities = []
        for x_ in x:
            _c, _q = self._parse( regex, x_ )
            quantities.append( _q )
            components.append( _c )
            if len( _c ) > max_components:
                max_components = len( _c )

        def _cast_array( values, empty_val, dtype ) -> NDArray:
            arr = np.full(
                (len( values ), max_components),
                empty_val,
                dtype=dtype
            )
            for i, val in enumerate( values ):
                n = len( val )
                arr[i, :n] = val
            return arr

        components = _cast_array( components, 'none', 'O' )

        if not self.parse_quantities:
            return components

        quantities = _cast_array( quantities, 0, 'float' )
        # normalise quantities
        quantities = quantities / quantities.sum( axis=1, keepdims=True )

        return components, quantities

    def _join( self, components: Iterable, quantities: Iterable ) -> str:
        # deal with single component first
        if len( components ) == 1:
            return components[0]

        # remove 'none' entries
        components = filter( lambda x: x != 'none', components )
        quantities = filter( None, quantities )

        # combine into string
        c = self.delimit.join( components )
        q = self.delimit.join( quantities )
        return f'{c} ({q})'

    def inverse_map( self, components: ArrayLike, quantities: ArrayLike ) -> List[str]:
        return [ self._join( c, q ) for c, q in zip( components, quantities ) ]


class BlendEncoder( BaseEstimator, TransformerMixin ):
    """
    Encodes strings describing mixtures of arbitary components.

    A blend describes a mixture of two or more components in specific quantities.
    A binary mixture of components is represented as a string in the format
    `'A:B x:y'` where A and B are the two component names and x:y give the ratio
    of the respective components; higher order mixtures are similarly
    represented by extending the delimiter pattern e.g. `'A:B:C x:y:z'` for a
    ternary mixture.

    Naively encoding each unique blend as a category discards potentially
    important relational information; two blends `'A:B 8:2'` and `'A:B 9:1'` may
    have similar properties whereas `'C:D 8:2'` could be entirely different.
    Blends with varying component types should not be encoded ordinally and a
    one-hot encoding could lead to a large increase in the number of features.

    This encoder parses each string to extract each component name and
    optionally it's quantity information. A one-hot encoding is created based on
    the unique collection of components found with the resultant vectors scaled
    to represent the component ratios.

    Parameters
    ----------
    delimit : str, default = ':'
        Character to separate component parts of the blend.
    case_sensitive : bool, default = False
        Perform case sensitive matching or not; by default matching is
        case insensitive.
    parse_quantities : bool, deafult = True
        Extract quantity information and scale vectors accordingly.
    validate : bool, default = True
        Check that blend quantities sum to 100%. Only relevent when
        `parse_quantities=True`.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X`
        has feature names that are all strings.
    encoders_ : list of length `n_features_in_`
        The fitted OneHotEncoder for each feature.
    components_ : list of length `n_features_in_`
        The found component names (as an array) for each feature.

    Examples
    --------
    Given a dataset of arbitary blends the encoder finds the unique components
    and quantities:

    >>> X = np.array([['A:B 1:1'], ['A:C 25.5:74.5'], ['B:C:A 1:1:2'], ['C']])
    >>> enc = BlendEncoder().fit(X)
    >>> enc.components_
    [array(['A', 'B', 'C'], dtype=object)]
    >>> enc.transform(X)
    array([[0.5  , 0.5  , 0.   ],
          [0.255, 0.   , 0.745],
          [0.5  , 0.25 , 0.25 ],
          [0.   , 0.   , 1.   ]])

    Feature names are generated as per scikit-learns's OneHotEncoder:

    >>> enc.get_feature_names_out()
    ['x0_A', 'x0_B', 'x0_C']

    Component ratios can be ignored, in which case a one-hot encoding of the
    unique components is given:

    >>> enc = BlendEncoder(parse_quantities=False)
    >>> enc.fit_transform(X)
    array([[1., 1., 0.],
          [1., 0., 1.],
          [1., 1., 1.],
          [0., 0., 1.]])

    Blends without component ratios are similarly supported:

    >>> X = np.array([['A:B'], ['A:C:D'], ['B'], ['B:D'], ['A:D']])
    >>> enc = BlendEncoder(parse_quantities=False)
    >>> enc.fit_transform(X)
    array([[1., 1., 0., 0.],
          [1., 0., 1., 1.],
          [0., 1., 0., 0.],
          [0., 1., 0., 1.],
          [1., 0., 0., 1.]])

    Notes
    -----
    The regex for matching components matches any word-type character plus
    characters from the set `-()[]` at the start of the string. Component
    quantities are expected as integer or decimal numbers at the end of the
    string. Components and quantities must be separated by whitespace. If no
    delimiter is present in the string it is assumed to be a single component
    blend.

    """

    def __init__(
            self, *,
            delimit: str = ':',
            case_sensitive: bool = False,
            parse_quantities: bool = True,
            validate: bool = True,
            ):
        self.delimit = delimit
        self.case_sensitive = case_sensitive
        self.parse_quantities = parse_quantities
        self.validate = validate

    def _check_X( self, X: ArrayLike, *, reset=False ):
        # do feature checks first
        self._check_n_features( X, reset=reset )
        self._check_feature_names( X, reset=reset )

        X = check_array(
            X,
            dtype=None,
            estimator="BlendEncoder"
            )
        return np.array( X )

    def _fit_transform( self, X: ArrayLike, transform: bool ):
        # X = np.atleast_2d( X )
        X = self._check_X( X, reset=True )
        encoders = []
        components = []
        transformed = []
        for i in range( self.n_features_in_ ):
            _components, _quantities = self._parse_1d( X[:, i] )
            _one_hot = self._fit_1d( _components )
            encoders.append( _one_hot )
            components.append( _one_hot.categories_[0] )

            if transform:
                transformed.append( self._transform_1d( _components, _quantities, _one_hot ) )

        self.encoders_ = encoders
        self.components_ = components
        if not transform:
            return self
        return np.concatenate( transformed, axis=1 )

    def fit( self, X, y=None ):
        """
        Fit BlendEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the blend components of each feature.
        y : None
            Ignored. This parameter exists only for scikit-learn compatibility.

        Returns
        -------
        self
            Fitted encoder.

        """
        return self._fit_transform( X, transform=False )

    def fit_transform( self, X, y=None ):
        """
        Fit BlendEncoder to X, then transform X.

        Equivalent to fit(X).transform(X) but more convenient and efficient
        (strings only have to be parsed once).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
        y : None
            Ignored. This parameter exists only for scikit-learn compatibility.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input. If `sparse=True`, a sparse matrix will be
            returned.

        """
        return self._fit_transform( X, transform=True )

    def transform( self, X ):
        """
        Transform X using modified one-hot encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input. If `sparse=True`, a sparse matrix will be
            returned.

        """
        check_is_fitted( self )
        X = self._check_X( X, reset=False )

        transformed = []
        for i, enc in enumerate( self.encoders_ ):
            components, quantities = self._parse_1d( X[:, i] )
            transformed.append( self._transform_1d( components, quantities, enc ) )
        return np.concatenate( transformed, axis=1 )

    def get_feature_names_out( self, input_features=None ) -> List[ str ]:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.
            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then names are generated: `[x0, x1, ..., x(n_features_in_)]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        List of strings of length `n_features_in_`
            Transformed feature names.

        """
        check_is_fitted( self )
        input_features = _check_feature_names_in( self, input_features )

        # feature_names_in = ( f'x{i}' for i in range( self.n_features_in_ ) )
        feature_components = ( enc.categories_[0] for enc in self.encoders_ )
        out = []
        for f_name, components in zip( input_features, feature_components ):
            for c_name in components:
                out.append( f'{f_name}_{c_name}' )
        return out

    def inverse_transform( self, Xt: ArrayLike ) -> NDArray:
        """
        Convert the data back to the original representation.

        When unknown components are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this component.

        Parameters
        ----------
        Xt : {array-like, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Inverse transformed array.

        Raises
        ------
        ValueError
            When the shape of Xt is not correct

        """
        check_is_fitted(self)
        Xt = check_array(Xt, accept_sparse="csr")

        Xt = np.array( Xt )
        n_samples, n_features = Xt.shape
        n_transformed_features = sum( len( cats ) for cats in self.components_ )

        if n_features != n_transformed_features:
            raise ValueError(
                "Shape of the passed X data is not correct. Expected {0} columns, got {1}.".format(
                    n_transformed_features,
                    n_features
                )
            )

        start_idx = 0
        X = []
        for enc in self.encoders_:
            stop_idx = start_idx + len( enc.categories_[0] )
            _Xt = Xt[:, start_idx:stop_idx]
            X.append( self._inverse_transform_1d( _Xt, enc.categories_[0] ) )
            start_idx = stop_idx
        X = np.concatenate( X, axis=1 )
        return X

    def _parse_1d( self, x ):
        parser = _BlendParser(
            self.delimit,
            case_sensitive=self.case_sensitive,
            parse_quantities=self.parse_quantities
            )
        if self.parse_quantities:
            return parser.map( x )
        # construct a dummy array of quantities
        _components = parser.map( x )
        _quantities = np.ones_like( _components, dtype=np.float64 )
        return _components, _quantities

    def _fit_1d( self, components ):
        categories = np.unique( components.flatten() )
        categories = categories[ categories != 'none' ]
        max_components = components.shape[1]

        one_hot = OneHotEncoder(
            categories=[categories]*max_components,
            sparse=False,
            handle_unknown='ignore',  # drops out 'none' entries
            )
        one_hot.fit( components )
        return one_hot

    def _transform_1d( self, components, quantities, encoder ):
        n_categories = len( encoder.categories_[0] )
        n_features = encoder.n_features_in_
        if components.shape[1] < n_features:
            components = np.pad( components, ((0, 0), (0, 1)), constant_values='none' )
            quantities = np.pad( quantities, ((0, 0), (0, 1)), constant_values=0 )
        encoded_2d = encoder.transform( components )

        _zip = zip(
            range( 0, encoded_2d.shape[1], n_categories ),
            quantities.T
            )
        encoded_stack = np.stack(
            [ q.reshape(-1, 1) * encoded_2d[:, i:i+n_categories] for i, q in _zip ]
            ).sum( axis=0 )

        if self.parse_quantities and self.validate and not self._validate_mixture( encoded_stack ):
            raise ValueError( 'Found one or more blends that do not sum to 100%' )

        return encoded_stack

    @staticmethod
    def _validate_mixture( mix_array ):
        total = mix_array.sum( axis=1 )
        return np.all( total == 1 )

    def _inverse_transform_1d( self, X, categories ):
        m, n = X.shape
        if n != len( categories ):
            raise ValueError( f"Shape {X.shape} of `X` did not match the given number of categories {len( categories )}" )

        gen = ( [ (name, val * 100) for name, val in zip( categories, x ) if val ] for x in X )
        return np.array( list( self._join( gen ) ) ).reshape( -1, 1 )

    @staticmethod
    def _join( iter ):
        for i in iter:
            if len( i ) == 0:
                yield None
            elif len( i ) == 1:
                yield i[0][0]
            else:
                components, quantities = zip( *sorted( i, key=lambda T: T[1], reverse=True ) )
                components = ':'.join( components )
                quantities = ':'.join( ( '{:.3g}'.format( q ) for q in quantities ) )
                yield components + ' ' + quantities


