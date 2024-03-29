navisML
=======

`navisML` provides an easy interface between `navis` module for neurons analysis and `scikit-learn` for machine learning.
For every neuron object (mesh and skeleton) from a neuronlist it enables you to extract scalar information to use as a feature in a ML modelling.

## Installation

```
$ pip install git+https://github.com/dokato/navisML
```

## Example

```python
import navis
from navisML.extractor import NeuralFeatures
from sklearn.cluster import KMeans

neurons = navis.read_swc("path/to/data.zip", read_meta=True)

nrnfeats = NeuralFeatures({
    'upstream' : 'upstream',
    'downstream' : 'downstream',
    'has_soma' : 'has_soma',
    'custom_feature' : custom_function
})

X = nrnfeats.fit_transform(neurons)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

```

## References

- [Navis](https://github.com/navis-org/navis)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn/)
