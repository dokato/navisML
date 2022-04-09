import os
import navis

__all__ = ['load_train', 'load_test']

fp = os.path.dirname(__file__)

data_path = os.path.join(fp, 'data')

def load_train():
    """
    Load example trainings Kenyon Cell neurons.
    """
    neurons = navis.read_swc(os.path.join(data_path,
                                          "kc_train_swc_N.zip"),
                             read_meta=True,
                             parallel=False)
    return neurons

def load_test():
    """
    Load example test Kenyon Cell neurons.
    """
    neurons = navis.read_swc(os.path.join(data_path,
                                          "kc_test_swc_N.zip"),
                             read_meta=True,
                             parallel=False)
    return neurons
