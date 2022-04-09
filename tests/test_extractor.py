from navisML.extractor import _is_numeric, _get_dtype, NeuralFeatures

import pytest
import navis
import numpy as np

@pytest.fixture
def neurons():
    "An example navis neuronlist."
    nl = navis.example_neurons(5)
    return nl

def test_get_dtype():
    assert _get_dtype(1) is np.dtype('int64')

def test_is_numeric():
    assert _is_numeric(np.array([33]).dtype) == True
    assert _is_numeric(np.array([True]).dtype) == False

def test_NF_features_parsing():
    nf = NeuralFeatures(['abc', 'def'])
    assert type(nf.features) is dict

def test_NF_features_input_list(neurons):
    nf = NeuralFeatures(['n_nodes', 'n_leafs'])
    nf.fit(neurons)
    assert len(nf._feature_types) == 2

def test_NF_features_input_dict(neurons):
    nf = NeuralFeatures({'a' : 'n_nodes', 'b' : 'n_leafs'})
    nf.fit(neurons)
    assert 'a' in nf._feature_types.keys()

def test_NF_features_input_function(neurons):
    f = lambda n: n.n_connectors
    nf = NeuralFeatures({'a' : 'n_nodes', 'b' : 'n_leafs', 'f' : f})
    res = nf.fit_transform(neurons)
    assert np.all(res.f == neurons.summary().n_connectors)

def test_NF_fit_transform(neurons):
    nf = NeuralFeatures({'a' : 'n_nodes', 'b' : 'n_leafs'})
    res = nf.fit_transform(neurons)
    assert res.shape == (5, 2)
    assert res.columns.to_list() == ['a', 'b']
