import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

def z_calc(y, kdir, krev, v_f, v_b):
    """
    Calculates reversibility of all elementary reactions present in the reaction mechanism.        
    Args:
        y(nparray): Steady state surface coverage at desired reaction conditions.
        kdir,krev(list): Kinetic constants.       
    Returns:
        List with reversibility of elementary reactions.
    """
    rd = np.zeros(len(kdir))
    ri = np.zeros(len(krev))
    for reaction in range(len(rd)):
        # Forward reaction rate
        rd[reaction] = kdir[reaction] * np.prod(y ** v_f[:, reaction])
        # Backward reaction rate
        ri[reaction] = krev[reaction] * np.prod(y ** v_b[:, reaction])
    return ri/rd

def calc_eapp(temperature_vector, reaction_rate_vector):
    """
    Function that calculates the apparent activation energy of a global reaction.
    A linear regression is performed to extract the output.
    Args:
        temperature_vector(nparray): Array containing the studied temperature range in Kelvin
        reaction_rate_vector(nparray): Array containing the reaction rate at different temperatures            
    Returns:
        Apparent reaction energy in kJ/mol in the temperature range of interest.      
    """
    lm = LinearRegression()
    x = pd.DataFrame(1 / temperature_vector)
    y = pd.DataFrame(np.log(reaction_rate_vector))
    reg = lm.fit(x, y)
    Eapp = -(8.31439 / 1000.0) * reg.coef_[0, 0]  # kJ/mol
    R2 = reg.score(x, y)
    return Eapp, R2

def calc_reac_order(partial_pressure, reaction_rate):
        """
        Args:
            partial_pressure(nparray): Partial pressure of the gas species [Pa]
            reaction_rate(nparray): Reaction rate [1/s]            
        Returns:
            Apparent reaction order with respect to the selected species
        """
        lm = LinearRegression()
        x = pd.DataFrame(np.log(partial_pressure))
        y = pd.DataFrame(np.log(reaction_rate))
        reg = lm.fit(x, y)
        napp = reg.coef_[0, 0]
        R2 = reg.score(x, y)
        return napp, R2

def stoic_forward(matrix):
    """
    Filter function for the stoichiometric matrix.
    Negative elements are considered and changed of sign in order to 
    compute the direct reaction rates.
    """
    mat = np.zeros([matrix.shape[0], matrix.shape[1]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if matrix[i][j] < 0:
                mat[i][j] = - matrix[i][j]
    return mat

def stoic_backward(matrix):
    """
    Filter function for the stoichiometric matrix.
    Positive elements are considered and kept in order to compute 
    the reverse reaction rates.
    """
    mat = np.zeros([matrix.shape[0], matrix.shape[1]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if matrix[i][j] > 0:
                mat[i][j] = matrix[i][j]
    return mat