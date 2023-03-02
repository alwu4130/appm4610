import numpy as np
import math
import matplotlib.pyplot as plt


def driver():
# example 1
#     y = lambda t: (t+1)**2-0.5*np.exp(t)
#     ya = 0.5
#     a = 0
#     b = 2
#     h = 0.2

#example 2
     y = lambda t: (((t**2) + (2*t) + 6)** 0.5) - 1
     ya = 2
     a = 1
     b = 1+(1e-5)
     h = 10**np.linspace(-5, -10, 6)

     #N = int((b-a)/h)
     yapp1 = np.zeros(len(h))
     t = np.zeros(len(h))
     for i in range(len(h)):
        [yapp1[i],t[i]] = explicit_euler(a,b,h[i],ya)
    
     print('yapp1 = ', yapp1)
     print('t', t)
     
     err1 = np.zeros(len(h))
     WHY = np.zeros(len(h))
     for j in range(0,len(h)):
        WHY[j] = y(t[j])
        err1[j] = abs(y(t[j])-yapp1[j])
     print('y = ', WHY)    
     print('err for explicit euler:', err1) 
     
    #  plt.plot(t,yapp1,label = 'explicit')
    #  plt.plot(t,y(t),label = 'exact')
    #  plt.xlabel('t')
    #  plt.ylabel('approx')
    #  plt.legend(loc = 'upper left')
    #  plt.show()

     plt.loglog(h,err1,label = 'explicit')
     plt.yscale('log')
     plt.xlabel('h')
     plt.ylabel('absolute err')
     plt.legend(loc = 'upper left')
     plt.title("Absolute Error vs. Time Step h")
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
        ftmp = eval_f(tj,yapp[jj-1])
        yapp[jj] = yapp[jj-1]+h*ftmp

     return (yapp[-1],t[-1])

     
def  eval_f(t,y):
# example 1
#     f = y -t**2+1

# example 2
     f = (1+t)/(1+y)

     return f     
     
driver()                    
     
