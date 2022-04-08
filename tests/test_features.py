from navisML.features import *

import pytest

@pytest.fixture
def neuron():
    "An example navis neuron."
    n = navis.example_neurons(1)
    return n

def test_get_cbf_length(neuron):
    expected_value = 6785.42
    assert abs(get_cbf_length(neuron) - expected_value) < 0.1

def test_get_root_radius(neuron):
    expected_value = 10
    assert get_root_radius(neuron) == 10

def test_get_average_radius(neuron):
    expected_value = 25.297
    assert abs(get_average_radius(neuron) - expected_value) < 0.1
