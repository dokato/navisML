import navis
from navisML.extractor import *
from navisML.features import *
from navisML import load_train, load_test
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

kc_train = load_train()
kc_test = load_test()

y_train = kc_train.summary(add_props=['cell_type']).cell_type.to_list()
y_test  = kc_test.summary(add_props=['cell_type']).cell_type.to_list()

nfeats = NeuralFeatures({
    'upstream' : 'upstream',
    'downstream' : 'downstream',
    'voxels' : 'voxels',
    'cellBodyFiber' : 'cellBodyFiber',
    'has_soma' : 'has_soma',
    'cbf_length' : get_cbf_length
})

x_train = nfeats.fit_transform(kc_train)
x_test = nfeats.transform(kc_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

clf =  DecisionTreeClassifier(max_depth=5)
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(f"Classification accuracy: {score:.3f}")
