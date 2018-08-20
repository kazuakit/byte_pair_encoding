# coding: utf-8

import pytest
import pathmagic # noqa
from byte_pair_encoding.encoder import Encoder


@pytest.fixture(scope='module')
def encoder():
    e = Encoder()
    e.train('ABCDCDABCDCDE')
    return e


def test_train(encoder):
    assert encoder.vocab is not None


def test_encode(encoder):
    s = 'ABCDCDABCDCDE'
    assert encoder.encode(s) == ['ABCDCD', 'ABCDCD', 'E']


def test_encode_unknown_seq(encoder):
    s = 'ABCDCDXXABCDCDE'
    assert encoder.encode(s) == ['ABCDCD', 'XX', 'ABCDCD', 'E']
