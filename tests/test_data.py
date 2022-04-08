from navisML.data import load_train, load_test

import pytest

def test_load_train():
    assert len(load_train()) == 300

def test_load_test():
    assert len(load_test()) == 75
