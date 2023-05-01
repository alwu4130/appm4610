import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sp


## PROBLEM 1
def main():
    # givens:
    a = 0
    b = 1
    N = 5
    N2 = 35

    [x, y] = FEM_solve(a, b, N)
    [x2, y2] = FEM_solve(a, b, N2)
    plt.plot(x, y, label = "N = 5")
    plt.plot(x2, y2, label = "N = 35")
    plt.legend()
    plt.title("Schrodinger Operator Approximation using FEM")
    plt.ylabel("u")
    plt.xlabel("x")
    plt.show()

    return
def FEM_solve(a, b, N):
    h = (b-a)/N
    xh = np.linspace(a, b, N) # number of elements in vector, size of matrix
    A = A_mat(N)
    b = b_vec(N)

    alpha = sp.linalg.spsolve(A, b)
    sol = evalBasis(alpha, N, xh[1:N+1])
    yapp=np.zeros(len(xh))
    yapp[1:N+1]=sol
    return [xh, yapp]

def A_mat(N):
    main_diag = np.zeros(N)
    for i in range(1,N+1):
       main_diag[i-1] = ((i*np.pi)**2)+2*(i%2)
    fill = [main_diag]
    diag_pos =[0]

    for i in range(1,int(N/2)+1):
        pos_entry = 2*i
        neg_entry = -(2*i)
        diag_pos.insert(0, neg_entry)
        diag_pos.append(pos_entry)

        diag = np.zeros(N-2*i)
        for j in range(len(diag)):
            diag[j] = (-1)**i*(1+(-1)**(j))

        fill.append(diag)
        fill.insert(0, diag)             
    A = sp.diags(fill,diag_pos,format='csc')

    return A


def b_vec(N):
    b = np.zeros(N)
    for i in range(1,N+1):
        b[i-1] = -(np.sqrt(2)/(i*np.pi))*(((-1)**i)-1)

    return b

def evalBasis(alpha,N,x):
    sol=np.zeros(N)
    for i in range(N):
        sol+=alpha[i]*np.sqrt(2)*np.sin((i+1)*np.pi*x)
    return sol
main()

## PROBLEM 2
def main2():

    # givens:
    a = 0
    b = 1
    alpha = 0
    beta = 0
    N = 32
    epsilon = 0.1
    epsilon2 = 0.25
    epsilon3 = 1

    [x, y] = FEM_solve(a, b, N, epsilon)
    [x2, y2] = FEM_solve(a, b, N, epsilon2)
    [x3, y3] = FEM_solve(a, b, N, epsilon3)

    # Plotting
    plt.plot(x, y, label = "epsilon = 0.1")
    plt.plot(x2, y2, label = "epsilon = 0.25")
    plt.plot(x3, y3, label = "epsilon = 1")
    plt.legend()
    plt.title("Reaction Diffusion with Various Epsilon Values")
    plt.ylabel("u")
    plt.xlabel("x")
    plt.show()

    # call function lmao
    return

def FEM_solve(a, b, N, epsilon):
    h = (b-a)/N
    xh = np.linspace(a, b, N+1) # number of elements in vector, size of matrix
    A = A_matrix(h, N-1, epsilon)
    b = b_vector(h, N-1)

    x_alpha = sp.linalg.spsolve(A, b)
    sol = np.zeros(len(xh))
    sol[1:N] = x_alpha

    return [xh, sol]

def A_matrix(h, N, epsilon):
    main_diag = ((2*h/3)+(2*epsilon/h))*np.ones(N)
    side_diag = ((h/6)-(epsilon/h))*np.ones(N-1)

    fill = [side_diag, main_diag, side_diag]
    diag_pos = [-1, 0, 1]
    A = sp.diags(fill,diag_pos,format='csc')

    return A


def b_vector(h, N):
    b = h*np.ones(N)

    return b
main2()