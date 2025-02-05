import enum

import numpy as np

inv_pi = 1/np.pi
inv_2_pi = 0.5*inv_pi
inv_4_pi = 0.25*inv_pi
pi_over_2 = np.pi/2
pi_over_4 = 0.5*pi_over_2
EPSILON = 0.000001

ZEROS = np.zeros((3), dtype=np.float64)
ONES = np.ones((3), dtype=np.float64)

BLUE = np.array([.25, .25, .75], dtype=np.float64)