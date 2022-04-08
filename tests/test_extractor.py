from navisML.extractor import _is_numeric, _get_dtype, NeuralFeatures

import pytest
import numpy as np

@pytest.fixture
def neurons():
    "An example navis neuronlist."
    nl = navis.example_neurons(10)
    return nl

def test_get_dtype():
    assert _get_dtype(1) is np.dtype('int64')

def test_is_numeric():
    assert _is_numeric(np.array([33]).dtype) == True
    assert _is_numeric(np.array([True]).dtype) == False

def test_NF_features_parsing():
    nf = NeuralFeatures(['abc', 'def'])
    assert type(nf.features) is dict
