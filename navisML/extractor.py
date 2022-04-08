import warnings

import navis
import numpy as np
import pandas as pd
from typing import Union, Any, NoReturn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import FitFailedWarning
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

def _verify_feature_value(x : Any, feature_name : str = "") -> NoReturn:
    """Warns when the feature value is None

    Args:
        x (Any): any feature value
        feature_name (str, optional): _description_. Defaults to "".
    """
    if x is None:
        warnings.warn(f'navisML Warning:: some feature values {feature_name} are None')

def _get_dtype(x : Any) -> np.typing.DTypeLike:
    """Get dtype of an object

    Args:
        x (Any): any object

    Returns:
        np.typing.DTypeLike: numpy dtype
    """
    return np.array(x).dtype

def _is_numeric(dtype : np.typing.DTypeLike) -> bool:
    """Is array a numeric type?

    Args:
        dtype (np.typing.DTypeLike): array-like object to test

    Returns:
        bool: True if *x* has numeric dtype (int, float), or False
    """
    return np.issubdtype(dtype, np.number)


class NeuralFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features : Union[dict, list], names = None):
        self.features = features
        self._features_to_dict(names)
        self._feature_types = None
        self._feature_encoders = {}

    def _features_to_dict(self, names : Union[list, None]):
        """Make sure that features are a dictionary.

        Args:
            names (Union[list, None]): feature names.

        Raises:
            ValueError: When illegal *names* are passed.
        """
        if isinstance(self.features, dict) and not names is None:
            raise ValueError("*names* arguments cannot be used with *features* dict")
        if isinstance(self.features, list):
            if names and len(self.features) != len(names):
                raise ValueError("List of names must be the same length as list of features.")
            fname = [f"X{i}" for i in range(len(self.features))]
            self.features = dict(zip(fname, self.features))

    def _check_features(self, neurons : navis.NeuronList):
        """Check features.

        Check if features exist for all the *neurons*.

        Args:
            neurons (navis.NeuronList): neuron's list to verify feature attributes.

        Raises:
            AttributeError: If wrong feature type passed

        """
        self._feature_types = {}
        for feat_name in self.features:
            if isinstance(self.features[feat_name], str):
                if self.features[feat_name] in neurons.summary().columns:
                    self._check_features_type(feat_name, neurons.summary()[self.features[feat_name]])
                    continue
                for nrn in neurons:
                    x = getattr(nrn, self.features[feat_name])
                    self._check_features_type(feat_name, x)
                continue
            if not  callable(self.features[feat_name]):
                raise AttributeError("Wrong feature type. Allowed types: str, Callable.")
        self._sweep_features_type()

    def _check_features_type(self, feature_name : str, feature : Any):
        if not feature_name in self._feature_types:
            self._feature_types[feature_name] = set([_get_dtype(feature)])
        else:
            self._feature_types[feature_name].add(_get_dtype(feature))
        if len(self._feature_types[feature_name]) > 1 and \
            not np.all([_is_numeric(x) for x in  self._feature_types[feature_name]]):
            raise TypeError((f"Inconsistent feature types in {feature_name} : "
                             f"{self._feature_types[feature_name]}"))

    def _sweep_features_type(self):
        for feature_name in self._feature_types:
            self._feature_types[feature_name] = _is_numeric(list(self._feature_types[feature_name])[0])
            if not self._feature_types[feature_name]:
                self._feature_encoders[feature_name] = LabelEncoder()

    def _encode_features(self, X : pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features to numeric with sklearn.preprocessing._label.LabelEncoder.

        Args:
            X (pd.DataFrame): pandas dataframe with raw features

        Raises:
            NotFittedError: if executed before fitting

        Returns:
            pd.DataFrame: data frame with all numeric columns
        """
        check_is_fitted(self, '_feature_types')
        Xenc = X.copy()
        for feature_name in self._feature_types:
            if not self._feature_types[feature_name]:
                Xenc[feature_name] = self._feature_encoders[feature_name].fit_transform(X[feature_name])
        return Xenc

    def decode_feature(self, feature_name : str, values : np.typing.ArrayLike) -> np.typing.ArrayLike:
        """Decode numeric features.

        Args:
            feature_name (str): feature_name
            values (np.typing.ArrayLike): values to decode

        Raises:
            ValueError: when trying to decode numerical feature
            NotFittedError: when called before fitting

        Returns:
            np.typing.ArrayLike: sequence with decoded feature
        """
        check_is_fitted(self, '_feature_types')
        if not feature_name in self._feature_encoders:
            raise ValueError(f"Looks like {feature_name} wasn't encoded. Is it numeric?")
        else:
            self._feature_encoders.transform(values)

    def _extract_features(self, neurons : navis.NeuronList, encode : bool = True) -> pd.DataFrame:
        """Extract features

        Args:
            neurons (navis.NeuronList): neuron's list
            encode (bool, optional): whether to encode categorical features to numeric
                                     (default True)

        Returns:
            pd.DataFrame: data frame with scalar neuron's features
        """
        X = np.empty((len(neurons), len(self.features)), dtype=object)
        for en, nrn in enumerate(neurons):
            for ef, feat_name in enumerate(self.features):
                if isinstance(self.features[feat_name], str):
                        X[en, ef] = getattr(nrn, self.features[feat_name])
                elif callable(self.features[feat_name]):
                        X[en, ef] = self.features[feat_name](nrn)
                _verify_feature_value(X[en, ef], feat_name)
        X = pd.DataFrame(X)
        X.columns = list(self.features.keys())
        if encode:
            X = self._encode_features(X)
        if getattr(neurons, 'name', None) is None:
            X.index = neurons.name
        return X

    def fit(self, neurons : navis.NeuronList):
        """Fit feature extractors.

        Args:
            neurons (navis.NeuronList): neurons list

        Returns:
            pd.DataFrame: data frame with scalar neuron's features
        """
        self._check_features(neurons)
        self.names = list(self.features.keys())
        return self

    def transform(self, neurons : navis.NeuronList):
        """Extract features

        Args:
            neurons (navis.NeuronList): neurons list

        Returns:
            pd.DataFrame: data frame with scalar neuron's features
        """
        if self._feature_types is None:
            warning.warn("Fitting features first.")
            self.fit(neurons)
        return self._extract_features(neurons)

    def fit_transform(self, neurons : navis.NeuronList):
        """Fit feature extractor and extract features.

        Args:
            neurons (navis.NeuronList): neurons list

        Returns:
            pd.DataFrame: data frame with scalar neuron's features
        """
        self.fit(neurons)
        return self.transform(neurons)
