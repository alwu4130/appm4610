import numpy as np
import matplotlib.pyplot as plt

def c1(f, x0, h):
    C1 = (f(x0+h)  - (2*f(x0)) + f(x0-h) )/(h**2)
    return C1


def c2(f, x0, h):
    C2 = (-f(x0 + (2*h)) + (16*f(x0+h)) - (30*f(x0)) + (16*f(x0-h)) - (f(x0 - (2*h))))/(12*(h**2))
    return C2


f = lambda x: np.exp(x)
x0 = (7/8)*np.pi
h = 10**(-np.linspace(0,16,17))
C1 = np.zeros(len(h))
C2 = np.zeros(len(h))
for i in range(len(h)):
    C1[i] = abs(f(x0)-c1(f, x0, h[i]))
    C2[i] = abs(f(x0)-c2(f, x0, h[i]))

print(C1)

plt.loglog(h, C1, label = 'C1 approx.')
plt.loglog(h, C2, label = 'C2 approx.')
plt.legend(loc="upper right")
plt.xlabel("h")
plt.ylabel("Error")
plt.title('Error of 2nd Derivative Approximations of e^x at x0 = 'r'$(7/8)\pi$' )
plt.show()