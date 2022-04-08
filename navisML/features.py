import navis
import numpy as np
from typing import Union

def get_cbf_length(neuron : Union[navis.TreeNeuron, navis.MeshNeuron]):
    "Compute cell body fiber length of a *neuron*"
    return navis.cell_body_fiber(neuron).cable_length

def get_root_radius(neuron : Union[navis.TreeNeuron, navis.MeshNeuron]):
    "Compute root (not always soma) radius of a *neuron*"
    return float(neuron.nodes[neuron.nodes.type == 'root'].radius)

def get_average_radius(neuron : Union[navis.TreeNeuron, navis.MeshNeuron]):
    "Compute average radius of a *neuron*"
    return np.mean(neuron.nodes.radius)

