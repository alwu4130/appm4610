import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def driver():

     y = lambda t: -1/t
    
     a = 1
     ya = -1 #initial value
     b = 2
     h = 2 ** (-1 * np.linspace(1, 16, 16))
     del_t = 2**-8
    
    #  N = int((b-a)/h)
    
    #  [yapp1,t] = explicit_euler(a,b,h,ya)
    #  [yapp2,t] = Taylor_2nd(a,b,h,ya)
    #  tVec = np.linspace(a, b, len(yapp2))

     yapp1 = np.zeros(len(h))
     t1 = np.zeros(len(h))
     yapp2 = np.zeros(len(h))
     t2 = np.zeros(len(h))
     yapp3 = np.zeros(len(h))
     t3 = np.zeros(len(h))

     for i in range(len(h)):
        yapp1[i],t1[i]=Taylor_2nd(a,b,h[i],ya)
        yapp2[i],t2[i]=Taylor_2nd_backward(a,b,h[i],ya, del_t)
        yapp3[i],t3[i]=Taylor_2nd_center(a,b,h[i],ya, del_t)
        

    # # linear interpolation:
    #  p = interp1d(tVec, yapp2)
    #  y_hat_a = p(1.978)
    #  y_hat_e = y(1.978)
    #  print("interp approx y = ", y_hat_a)
    #  print("interp exact y = ", y_hat_e)
    #  print("error:", abs(y_hat_e-y_hat_a))
    
    #  err1 = np.zeros(N+1)
     err1 = np.zeros(len(h))
     err2 = np.zeros(len(h))
     err3 = np.zeros(len(h))
     
     for j in range(0,len(h)):
        # err1[j] = abs(y(t[j])-yapp1[j])
        err1[j] = abs(y(2)-yapp1[j])
        err2[j] = abs(y(2)-yapp2[j])
        err3[j] = abs(y(2)-yapp3[j])
    
    #  print('err for explicit euler:', err1) 
    #  print('err for 2nd order Taylor:', err2) 
    #  plt.plot(t,y(t),label = 'exact')
    # #  plt.plot(t,yapp1,label = 'Euler')
    #  plt.plot(t,yapp2,label = '2nd Order Taylor')
    #  plt.xlabel('t')
    #  plt.legend(loc = 'upper left')
    #  plt.show()

    #  plt.plot(t,err1,label = 'Euler')
     plt.loglog(h,err1,label = '2nd Order Taylor')
     plt.loglog(h,err2,label = 'Backward Difference')
     plt.loglog(h,err3,label = 'Centered Difference')
     #plt.yscale('log')
     plt.xlabel('h')
     plt.ylabel('absolute err')
     plt.legend(loc = 'upper left')
     plt.title('Solution Approximation')
     plt.show()
    
     return

# def explicit_euler(a,b,h,ya):

#      N = int((b-a)/h)
     
#      yapp = np.zeros(N+1)
#      t = np.zeros(N+1)
     
#      yapp[0] = ya
#      t[0] = a

#      for jj in range(1, N+1):
#         tj = a+(jj-1)*h
#         t[jj] = tj+h
#         ftmp = eval_f(tj,yapp[jj-1])
#         yapp[jj] = yapp[jj-1]+h*ftmp

#      return (yapp,t)
     
def Taylor_2nd(a,b,h,ya):

     N = int((b-a)/h)
     
     yapp = np.zeros(N+1)
     t = np.zeros(N+1)
     
     yapp[0] = ya
     t[0] = a

     for jj in range(1, N+1):
        tj = a+(jj-1)*h
        t[jj] = tj+h
        ftmp = eval_f(tj,yapp[jj-1])
        ft_tmp = eval_ft(tj,yapp[jj-1])
        yapp[jj] = yapp[jj-1]+h*(ftmp + h/2*ft_tmp)

     return (yapp[-1],t[-1])

def Taylor_2nd_backward(a,b,h,ya, del_t):

     N = int((b-a)/h)
     
     yapp = np.zeros(N+1)
     t = np.zeros(N+1)
     
     yapp[0] = ya
     t[0] = a

     for jj in range(1, N+1):
        tj = a+(jj-1)*h
        t[jj] = tj+h
        ftmp = eval_f(tj,yapp[jj-1])
        ft_tmp = eval_ft_22(tj,yapp[jj-1], del_t)
        yapp[jj] = yapp[jj-1]+h*(ftmp + h/2*ft_tmp)

     return (yapp[-1],t[-1])

def Taylor_2nd_center(a,b,h,ya, del_t):

     N = int((b-a)/h)
     
     yapp = np.zeros(N+1)
     t = np.zeros(N+1)
     
     yapp[0] = ya
     t[0] = a

     for jj in range(1, N+1):
        tj = a+(jj-1)*h
        t[jj] = tj+h
        ftmp = eval_f(tj,yapp[jj-1])
        ft_tmp = eval_ft_23(tj,yapp[jj-1], del_t)
        yapp[jj] = yapp[jj-1]+h*(ftmp + h/2*ft_tmp)

     return (yapp[-1],t[-1])

    
def eval_f(t,y):
#  evals the right hand side of the DE

    f = (1/(t**2)) - (y/t) - (y**2)
    return f

def eval_dfdt2(t,y,del_t):

    # backward difference
    dfdt = (eval_f(t, y)-eval_f(t-del_t, y))/del_t

    # centered difference
    # dfdt = (eval_f(t+del_t, y)-eval_f(t-del_t, y))/(2*del_t)

    return dfdt

def eval_dfdt3(t,y,del_t):

    # backward difference
    # dfdt = (eval_f(t, y)-eval_f(t-del_t, y))/del_t

    # centered difference
    dfdt = (eval_f(t+del_t, y)-eval_f(t-del_t, y))/(2*del_t)

    return dfdt

def eval_ft(t,y):
# evals Df/Dt = total derivative of f wrt t

    ft = (-2/(t**3)) + (y/(t**2)) + ((-1/t) - (2*y))*((1/(t**2)) - (y/t) - (y**2))
    # ft = eval_dfdt(t,y,del_t) + ((-1/t) - (2*y))*((1/(t**2)) - (y/t) - (y**2))

    return ft     

def eval_ft_22(t,y, del_t):
# evals Df/Dt = total derivative of f wrt t

    # ft = (-2/(t**3)) + (y/(t**2)) + ((-1/t) - (2*y))*((1/(t**2)) - (y/t) - (y**2))
    ft = eval_dfdt2(t,y,del_t) + ((-1/t) - (2*y))*((1/(t**2)) - (y/t) - (y**2))

    return ft  

def eval_ft_23(t,y, del_t):
# evals Df/Dt = total derivative of f wrt t

    # ft = (-2/(t**3)) + (y/(t**2)) + ((-1/t) - (2*y))*((1/(t**2)) - (y/t) - (y**2))
    ft = eval_dfdt3(t,y,del_t) + ((-1/t) - (2*y))*((1/(t**2)) - (y/t) - (y**2))

    return ft        
    
driver()
 
