import navis
import numpy as np
from typing import Union

def get_cbf_length(neuron : Union[navis.TreeNeuron, navis.MeshNeuron]) -> float:
    "Computes cell body fiber length of a *neuron*"
    return navis.cell_body_fiber(neuron).cable_length

def get_root_radius(neuron : Union[navis.TreeNeuron, navis.MeshNeuron]) -> float:
    "Computes root (not always soma) radius of a *neuron*"
    return float(neuron.nodes[neuron.nodes.type == 'root'].radius)

def get_average_radius(neuron : Union[navis.TreeNeuron, navis.MeshNeuron]) -> float:
    "Computes average radius of a *neuron*"
    return np.mean(neuron.nodes.radius)

def get_spine_radius(neuron : Union[navis.TreeNeuron, navis.MeshNeuron]) -> float:
    "Computes spine radius of a *neuron*"
    return float(np.mean(navis.longest_neurite(neuron).nodes.radius))

def get_spine_length(neuron : Union[navis.TreeNeuron, navis.MeshNeuron]) -> float:
    "Computes spine length of a *neuron*"
    return float(navis.longest_neurite(neuron).cable_length)
