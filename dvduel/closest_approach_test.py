"""Unit tests for closest_approach."""
import pytest
from pytest import approx
import numpy as np
from closest_approach import closest_approach


def test_perp2d():
    ### Setup ###
    p1 = np.array([1, 0, 0])
    p2 = np.array([-1, 0, 0])
    p3 = np.array([0, 1, 0])
    p4 = np.array([0, 2, 0])

    ### Action ###
    d = closest_approach(p1, p2, p3, p4)

    ### Verification ###
    assert d == approx(1.)


def test_parallel2d():
    ### Setup ###
    p1 = np.array([1, 0, 0])
    p2 = np.array([-1, 0, 0])
    p3 = np.array([1, 1, 0])
    p4 = np.array([-1, 1, 0])

    ### Action ###
    d = closest_approach(p1, p2, p3, p4)

    ### Verification ###
    assert d == approx(1.)


def test_intersect():
    ### Setup ###
    p1 = np.array([1, 0, 0])
    p2 = np.array([-1, 0, 0])
    p3 = np.array([0, 2, 0.])
    p4 = np.array([0, -2, 0.])

    ### Action ###
    d = closest_approach(p1, p2, p3, p4)

    ### Verification ###
    assert d == approx(0.)


def test_skew():
    ### Setup ###
    p1 = np.array([1, 0, 0])
    p2 = np.array([-1, 0, 0])
    p3 = np.array([0, 2, 1.])
    p4 = np.array([0, -2, 1.])

    ### Action ###
    d = closest_approach(p1, p2, p3, p4)

    ### Verification ###
    assert d == approx(1.)
