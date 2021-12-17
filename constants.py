# constants used in pymkm
import numpy as np

R = 8.31439                 # J/mol/K
N_AV = 6.02283E23           # 1/mol
K_BU = R / N_AV             # J/K
K_B = 8.61727E-5            # eV/K 
H = 4.1357E-15              # eV*s
cf = 96.49                  # (kJ/mol)/eV

m_dict = {'C':12.0107, 'H':1.00784, 'O':15.9994,
          'N':14.0067, 
          'P':30.97696,
          'S':32.065,
          'F':18.998403,
          'Cl':35.453,
          'Br':79.904,
          'I':126.9044,
          'He':4.002602,
          'Ne':20.1797,
          'Ar':39.948,
          'Kr':83.798,
          'Xe':131.293,
          'Rn':220.0,
          'Og':294.0} 

int_set = {'1','2','3','4','5','6','7','8','9','10'}
integers = {'1','2','3','4','5','6','7','8','9','0'}

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