import numpy as np
import math
import matplotlib.pyplot as plt


def driver():
     y = lambda t: -1/np.log(t+1) #exact solution
     
     a = 1
     ya = -(np.log(2))**-1 #initial vaue
     b = 2
     h = 0.1
     
     N = int((b-a)/h)
     
    #  [yapp1,t] = explicit_euler(a,b,h,ya)
    #  [yapp2,t] = Taylor_2nd(a,b,h,ya)
    #  [yapp3,t] = RK2(a,b,h,ya)
    #  [yapp4,t] = RK4(a,b,h,ya)
     [yapp5,t] = fourstepadams(a,b,h,ya)    
     [yapp6,t] = twostepadams(a,b,h,ya)    
     
     
     err1 = np.zeros(N+1)
     err2 = np.zeros(N+1)
    #  err3 = np.zeros(N+1)
    #  err4 = np.zeros(N+1)
     err5 = np.zeros(N+1)
     err6 = np.zeros(N+1)
     
     for j in range(0,N+1):
        # err1[j] = abs(y(t[j])-yapp1[j])
        # err2[j] = abs(y(t[j])-yapp2[j])
        # err3[j] = abs(y(t[j])-yapp3[j])
        # err4[j] = abs(y(t[j])-yapp4[j])
        err5[j] = abs(y(t[j])-yapp5[j])
        err6[j] = abs(y(t[j])-yapp6[j])
    
#     print('err for explicit euler:', err1) 
     
    #  plt.plot(t,yapp1,label = 'Euler')
    #  plt.plot(t,yapp2,label = '2nd Taylor')
    #  plt.plot(t,yapp3,label = 'RK2')
    #  plt.plot(t,yapp4,label = 'RK4')
     plt.plot(t,yapp5,label = '4 Step Adams-Bashforth')
     plt.plot(t,yapp6,label = '2 Step Adams-Bashforth')
     plt.plot(t,y(t),label = 'exact')
     plt.xlabel('t')
     plt.legend(loc = 'upper left')
     plt.title("Multi-Step Methods Comparison")
     plt.show()

    #  plt.plot(t,err1,label = 'Euler')
    #  plt.plot(t,err2,label = '2nd Taylor')
    #  plt.plot(t,err3,label = 'RK2')
    #  plt.plot(t,err4,label = 'RK4')
     plt.plot(t,err5,label = '4 Step Adams-Bashforth')
     plt.plot(t,err6,label = '2 Step Adams-Bashforth')
     plt.yscale('log')
     plt.xlabel('t')
     plt.ylabel('absolute err')
     plt.legend(loc = 'upper left')
     plt.title("Absolute Error for Multi-Step Methods")
     plt.show()
     
     return
     
def eval_f(t,y):
#     evaluate f(t,y)
     
     f = y**2/(1+t)
     return f
     
def eval_ft(t,y):
#      evaluate the total derivative of f
     
     ft = -y/(t**2)+1/t*(1+y/t)
     return ft
     
def fourstepadams(a,b,h,ya):

     N = int((b-a)/h)
     
     yapp = np.zeros(N+1)
     t = np.zeros(N+1)
 
#   initialize with rk4
     btmp = a+3*h
     [ytmp, ttmp] = RK4(a,btmp,h,ya)

#   copy the entries into yapp and t
     for j in range(0,4):
        yapp[j] = ytmp[j]
        t[j] = ttmp[j]     

# now time integration with Adams-Bashforth
     for j in range(4,N+1):
        t1 = a+(j-4)*h
        a1 = yapp[j-4]
        t2 = a+(j-3)*h
        a2 = yapp[j-3]
        t3 = a+(j-2)*h
        a3 = yapp[j-2]
        t4 = a+(j-1)*h
        a4 = yapp[j-1]
        t[j] = t4+h
        yapp[j] = yapp[j-1]+h/24*(55*eval_f(t4,a4)-59*eval_f(t3,a3)+
               37*eval_f(t2,a2)-9*eval_f(t1,a1))

     return (yapp,t)

def twostepadams(a,b,h,ya):

     N = int((b-a)/h)
     
     yapp = np.zeros(N+1)
     t = np.zeros(N+1)
 
#   initialize with rk4
     btmp = a+h
     [ytmp, ttmp] = RK4(a,btmp,h,ya)

#   copy the entries into yapp and t
     for j in range(0,2):
        yapp[j] = ytmp[j]
        t[j] = ttmp[j]     

# now time integration with Adams-Bashforth
     for j in range(2,N+1):
        t3 = a+(j-2)*h
        a3 = yapp[j-2]
        t4 = a+(j-1)*h
        a4 = yapp[j-1]
        t[j] = t4+h
        yapp[j] = yapp[j-1]+h/2*(3*eval_f(t4,a4)-eval_f(t3,a3))

     return (yapp,t)

# def fourstepadamsBAD(a,b,h,ya):

#      N = int((b-a)/h)
     
#      yapp = np.zeros(N+1)
#      t = np.zeros(N+1)
     
 
# #   initialize with rk4
#      btmp = a+3*h
#      [ytmp, ttmp] = RK2(a,btmp,h,ya)

# #   copy the entries into yapp and t
#      for j in range(0,4):
#         yapp[j] = ytmp[j]
#         t[j] = ttmp[j]     

# # now time integration with Adams-Bashforth
#      for j in range(4,N+1):
#         t1 = a+(j-4)*h
#         a1 = yapp[j-4]
#         t2 = a+(j-3)*h
#         a2 = yapp[j-3]
#         t3 = a+(j-2)*h
#         a3 = yapp[j-2]
#         t4 = a+(j-1)*h
#         a4 = yapp[j-1]
#         t[j] = t4+h
#         yapp[j] = yapp[j-1]+h/24*(55*eval_f(t4,a4)-59*eval_f(t3,a3)+
#                37*eval_f(t2,a2)-9*eval_f(t1,a1))
               

#      return (yapp,t)


     
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
     
# def Taylor_2nd(a,b,h,ya):

#      N = int((b-a)/h)
     
#      yapp = np.zeros(N+1)
#      t = np.zeros(N+1)
     
#      yapp[0] = ya
#      t[0] = a

#      for jj in range(1, N+1):
#         tj = a+(jj-1)*h
#         t[jj] = tj+h
#         ftmp = eval_f(tj,yapp[jj-1])
#         ft_tmp = eval_ft(tj,yapp[jj-1])
#         yapp[jj] = yapp[jj-1]+h*(ftmp + h/2*ft_tmp)

#      return (yapp,t)
     
# def RK2(a,b,h,ya):

#      N = int((b-a)/h)
     
#      yapp = np.zeros(N+1)
#      t = np.zeros(N+1)
     
#      yapp[0] = ya
#      t[0] = a

#      for jj in range(1, N+1):
#         tj = a+(jj-1)*h
#         t[jj] = tj+h
#         ftmp = eval_f(tj,yapp[jj-1])
#         ytmp = yapp[jj-1]+h/2*ftmp;
#         yapp[jj] = yapp[jj-1]+h*eval_f(tj+h/2,ytmp)

#      return (yapp,t)

def RK4(a,b,h,ya):

     N = int((b-a)/h)
     
     yapp = np.zeros(N+1)
     t = np.zeros(N+1)
     
     yapp[0] = ya
     t[0] = a

     for jj in range(1, N+1):
        tj = a+(jj-1)*h
        t[jj] = tj+h
        rk = yapp[jj-1]
        k1 = h*eval_f(tj,rk)
        k2= h*eval_f(tj+h/2,rk+0.5*k1)
        k3 = h*eval_f(tj+h/2,rk+1/2*k2)
        k4 = h*eval_f(tj+h,rk+k3)
        yapp[jj] = rk + 1/6*(k1+2*k2+2*k3+k4)

     return (yapp,t)

driver()


def f(t, y):
    return 1 - y

h = 0.1
w = [0, 1 - math.exp(-0.1)]

for i in range(2, 10):
    t = (i-2) + h
    w_i = 4 * w[i-1] - 3 * w[i-2] - 2 * h * f(t-h, w[i-2])
    w.append(w_i)

t_values = [i + h for i in range(10)]

plt.plot(t_values, w, linewidth=2, markersize=8)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Approximation of y'(t) w/ h = 0.1")


plt.show()

h = 0.01
w = [0, 1 - math.exp(-0.01)]
for i in range(2, 100):
    t = (i-2) + h
    w_i = 4 * w[i-1] - 3 * w[i-2] - 2 * h * f(t-h, w[i-2])
    w.append(w_i)

t_values = [i + h for i in range(100)]

plt.plot(t_values, w, linewidth=2, markersize=8)


plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Approximation of y'(t) w/ h = 0.01")
plt.show()

               
