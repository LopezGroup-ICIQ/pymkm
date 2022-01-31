from abc import abstractmethod
from mkm import MKM
from constants import *
from functions import *
from abc import ABC
import numpy as np
import math

def net_rate(y, kd, ki, v_matrix):
    """
    Returns the net reaction rate for each elementary reaction.
    Args:
        y(ndarray): surface coverage + partial pressures array [-/Pa].
        kd, ki(ndarray): kinetic constants of the direct/reverse steps.
    Returns:
        ndarray containing the net reaction rate for all the steps [1/s].
    """
    net_rate = np.zeros(len(kd))
    v_ff = stoic_forward(v_matrix)
    v_bb = stoic_backward(v_matrix)
    net_rate = kd * np.prod(y ** v_ff.T, axis=1) - ki * np.prod(y ** v_bb.T, axis=1)
    return net_rate

class ReactorModel(ABC):
    def __init__(self, name, params):
        self.name = name
        self._level = 1
        self.params = params
    @abstractmethod
    def ode(self):
        ...

    @abstractmethod
    def jacobian(self):
        ...
    
    @abstractmethod
    def termination_event(self):
        ...

class DifferentialPFR(ReactorModel):
    def ode(self, time, y, kd, ki, v_matrix):
        # Surface species
        dy = v_matrix @ net_rate(y, kd, ki, v_matrix)
        # Gas species
        dy[MKM.NC_sur:] = 0.0
        return dy

    def jacobian(self, time, y, kd, ki):
        pass

class DynamicCSTR(ReactorModel):
    def __init__(self, radius, length, Q, m_cat, s_bet, a_site):
        self.radius = radius
        self.length = length
        self.volume = (math.pi * radius **2) * length
        self.Q = Q
        self.tau = self.volume / self.Q
        self.m_cat = m_cat
        self.s_bet = s_bet
        self.a_site = a_site

    def ode(self, time, y, kd, ki, P_in, temperature):
        # Surface species
        dy = MKM.v_matrix @ net_rate(y, kd, ki)
        # Gas species
        dy[MKM.NC_sur:] *= (R * temperature / (N_AV * self.volume))
        dy[MKM.NC_sur:] *= (self.s_bet * self.m_cat / self.a_site)
        dy[MKM.NC_sur:] += (P_in - y[MKM.NC_sur:]) / self.tau
        return dy

              