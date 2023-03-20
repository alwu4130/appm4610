import numpy as np
import math
import matplotlib.pyplot as plt


def driver():
    y = lambda x: math.log(x)
    a = 1
    b = 2
    alpha = 0
    beta = math.log(2)

    tol = 1e-4
    N = 9
    N2 = 18
    M = 1000
    
    [x, w, ier] = nonlin_FD(a, b, alpha, beta, N, tol, M)
    [x2, w2, ier] = nonlin_FD(a, b, alpha, beta, N2, tol, M)
    print(w)
    print(w2)

    abs_err = np.zeros(len(w))
    rel_err = np.zeros(len(w))
    abs_err2 = np.zeros(len(w2))
    rel_err2 = np.zeros(len(w2))
    for i in range(len(w)):
        abs_err[i] = np.abs(y(x[i])-w[i])
        rel_err[i] = np.abs(abs_err[i]/y(x[i]))
        abs_err2[i] = np.abs(y(x2[i])-w2[i])
        rel_err2[i] = np.abs(abs_err2[i]/y(x2[i]))

    plt.plot(x, abs_err, label="abs error, N=9")
    plt.plot(x, rel_err, label="rel error, N=9")
    plt.plot(x2, abs_err2, label="abs error, N=18")
    plt.plot(x2, rel_err2, label="rel error, N=18")
    plt.yscale('log')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.legend(loc = 'best')
    plt.title("Nonlinear Finite Difference Error Comparison")
    plt.show()

def nonlin_FD(a, b, alpha, beta, N, tol, M):

    h = (b-a)/(N+1);

    w0 = alpha
    wN1 = beta

    w = np.zeros(N+2)
    w[0] = w0
    w[-1] = wN1
    for i in range(1, N+1):
        w[i] = alpha + (i*h*((beta-alpha)/(b-a)))

    k = 1
    A = np.zeros(N+2)
    B = np.zeros(N+2)
    C = np.zeros(N+2)
    D = np.zeros(N+2)
    while k <= M:
        x = a + h
        t = (w[2]-alpha)/(2*h)
        A[1] = 2 + (h**2) * eval_fy(x, w[1], t)
        B[1] = -1 + (h/2)*eval_fyp(x, w[1], t)
        D[1] = -(2*w[1] - w[2] - alpha + (h**2)*eval_f(x, w[1], t))

        for i in range(2, N):
            x = a + (i*h)
            t = (w[i+1]-w[i-1])/(2*h)
            A[i] = 2 + (h**2) * eval_fy(x, w[i], t)
            B[i] = -1 + (h/2)*eval_fyp(x, w[i], t)
            C[i] = -1 - (h/2)*eval_fyp(x, w[i], t)
            D[i] = -(2*w[i] - w[i+1] - w[i-1] + (h**2)*eval_f(x, w[i], t))

        x = b - h
        t = (beta - w[-2])/(2*h)
        A[-1] = (2 + (h**2) * eval_fy(x, w[-1], t))
        C[-1] = (-1 - (h/2)*eval_fyp(x, w[-1], t))
        D[-1] = (-(2*w[-1] - w[-2] - beta + (h**2)*eval_f(x, w[-1], t)))

        l = np.zeros(N+2)
        u = np.zeros(N+2)
        v = np.zeros(N+2)
        z = np.zeros(N+2)
        l[1] = A[1]
        u[1] = B[1]/A[1]
        z[1] = D[1]/l[1]

        for i in range(2,N):
            l[i] = A[i]- C[i]*u[i-1]
            u[i] = B[i]/l[i]
            z[i] = (D[i] - (C[i]*z[i-1]))/l[i]

        l[-1] = (A[-1] - C[-1]*u[-2])
        z[-1] = ((D[-1] - C[-1]*z[-2])/l[-1])

        v[-1] = z[-1]
        w[-1] = w[-1] + v[-1]
        for i in range(N, 1, -1):
            v[i] = z[i]- u[i]*v[i+1]
            w[i] = w[i] + v[i]
        
        if np.linalg.norm(v) <= tol:
            x = np.zeros(N+2)
            for i in range(0, N+2):
                x[i] = a + (i*h)
            print("Yay! Procedure was successful")
            ier = 0
            return x, w, ier
        
        k = k+1
    
    ier = 1
    w = []
    x = []
    print("Uh-oh max iterations exceeded sad")
    return x, w, ier
    

    


def eval_fyp(x, y, yp):
    f = 0
    return f

def eval_fy(x, y, yp):
    f = 2*np.exp(-2*y)
    return f

def eval_f(x, y, yp):
    f = -np.exp(-2*y)
    return f

driver()