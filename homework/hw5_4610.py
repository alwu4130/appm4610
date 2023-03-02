import numpy as np
import math
import matplotlib.pyplot as plt


def driver():

     y = lambda t: 2*np.exp(2*t) - np.exp(t) + np.exp(-t)
     ya = 2 # initial value
     a = 0
     b = 1
     h = 0.1

     N = int((b-a)/h)
     [yapp1,t] = explicit_euler(a,b,h,ya)
     
     print('yapp1 = ', yapp1)
     
     err1 = np.zeros(N+1)
     
     for j in range(0,N+1):
        err1[j] = abs(y(t[j])-yapp1[j])
        
     print('err for explicit euler:', err1) 
     
     plt.plot(t,yapp1,label = 'explicit')
     plt.plot(t,y(t),label = 'exact')
     plt.xlabel('t')
     plt.ylabel('approx')
     plt.legend(loc = 'upper left')
     plt.show()
     plt.plot(t,err1,label = 'explicit')
     plt.yscale('log')
     plt.xlabel('t')
     plt.ylabel('absolute err')
     plt.legend(loc = 'upper left')
     plt.show()
     return 
     
def explicit_euler(a,b,h,ya):

     N = int((b-a)/h)
     
     yapp = np.zeros(N+1)
     t = np.zeros(N+1)
     
     yapp[0] = ya
     t[0] = a

     for jj in range(1, N+1):
        tj = a+(jj-1)*h
        t[jj] = tj+h
        ftmp = eval_f(tj,yapp[jj-1]) #needs fixing
        yapp[jj] = yapp[jj-1]+h*ftmp

     return (yapp[-1],t[-1])

     
def  eval_f(t,u1, u2):

     u2p = 3*u2 -2*u1 + 6*np.exp(-t)
     u1p = u2

     return u2p, u1p     
     
driver()                    
     
