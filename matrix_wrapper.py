import os
import sys
import numpy as np

root_path = os.path.dirname(os.path.abspath(__file__))
process_path = root_path + '/params'

sys.path.append(process_path)

import matrix2py


def initialisemodel():
    print("Initializing model")
    matrix2py.initialisemodel(root_path+"/params/param_card.dat")

def smatrix(tensor: np.ndarray):
    result = np.array(
        [
            matrix2py.smatrix(row.T.tolist())
            for row in tensor
        ],
        dtype=np.float64
    )

    nan_ind = np.argwhere(np.isnan(result))
    for ind in nan_ind:
        print(f"first nan at {ind}")
        print(tensor[ind[0]])

        break

    return result
