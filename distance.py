import numpy as np

def row_distance(shape, a, b):
    # shape: the shape of the matrix on which we want to calculate row distance
    # a, b: coordinates to compare, row-major
    # everything is numpy arrays
    d_col = abs(a[1] - b[1])
    d_row = min(a[0] + b[0], 2 * shape[0] - a[0] - b[0])
    # Add 1 to all distances to avoid repeat sampling of same plot forever
    return abs(a[0] - b[0]) + 1 if a[1] == b[1] else sum((d_col, d_row)) + 1

def build_distance_matrix(shape, n_elements):
    D = np.zeros((n_elements, n_elements))
    for e_1 in range(n_elements):
        a = (e_1 // shape[1], e_1 % shape[1])
        for e_2 in range(n_elements):
            b = (e_2 // shape[1], e_2 % shape[1])
            d_e = row_distance(shape, a, b)
            D[e_1][e_2] = d_e
            D[e_2][e_1] = d_e
    return D
