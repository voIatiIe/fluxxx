import os
import sys
import numpy as np

root_path = os.path.dirname(os.path.abspath(__file__))
process_path = root_path + '/mg5/fortran_output/SubProcesses/P1_ddx_ddx_no_ag'

sys.path.append(process_path)

import matrix2py


def initialisemodel():
    print("Initializing model")
    matrix2py.initialisemodel(root_path+"/mg5/fortran_output/Cards/param_card.dat")

def smatrix(tensor: np.ndarray):
    print("Calculating matrix element")

    result = np.array(
        [
            matrix2py.smatrix(row.T.tolist())
            for row in tensor
        ],
        dtype=np.float32
    )

    return result
