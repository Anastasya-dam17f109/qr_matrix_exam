import numpy as np
import math

A_matr = np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 2.0, ], [4.0, 1.0, 3.0, ]])
n_size = 3
b_vec = np.array([[7.0], [8.0], [8.0]])

def Q_matr_genrator(A, size):
    Q =  np.arange(size*size, dtype = 'float64' ).reshape(size,size)
    alpha = np.zeros(size)
    for i in range(size):
        Q[:,i] = A[:,i].copy()
        for j  in range(i):
            alpha[j] = -np.dot(Q[:,i], Q[:,j])
        for j in range(i):
            Q[:, i] = Q[:, i]+ Q[:,j]*alpha[j]
        f_norm = np.sqrt(np.dot(Q[:,i], Q[:,i]))
        Q[:, i] = Q[:,i] /f_norm
    return Q

def solve_system(A,b, size):
    Q_matr = Q_matr_genrator(A, size)
    R_matr = np.dot(Q_matr.T, A)
    y = np.dot(Q_matr.T, b)
    x = np.linalg.solve(R_matr, y)
    return x

print(solve_system(A_matr, b_vec, n_size) )




