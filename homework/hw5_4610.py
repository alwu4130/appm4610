import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits import mplot3d 



def driver():

     ## Problem 4:
     # y = lambda t: -3 + 2*(1+np.exp(-2*t))**-1
     # ya = -2
     # a = 0
     # b = 3
     # hmax = 0.5
     # hmin = 0.05
     # tol = 10e-6

     # [yappRKF, t, stepRKF] = RKFehlberg(a,b,hmin,hmax,ya,tol)
     # [yappRK4_min, tmin] = RK4(a,b,0.05,ya)
     # [yappRK4_max, tmax] = RK4(a,b,0.5,ya)
     
     # print("Runge-Kutta-Fehlberg used",stepRKF,"steps.", "\n")
     # print("Error at t=3 for RK-Fehlberg:",np.abs(y(3)-yappRKF[-1]), "\n\n")

     # print("RK4 w/h=0.5 used",len(tmax),"steps.", "\n")
     # print("t=3 error was",np.abs(y(3)-yappRK4_max[-1]), "\n\n")

     # print("RK4 h=0.05 Needed",len(tmin),"steps.", "\n")
     # print("t=3 error was",np.abs(y(3)-yappRK4_min[-1]) , "\n\n")


     # errRKF = np.zeros(len(t))
     # errRK4_min = np.zeros(len(tmin))
     # errRK4_max = np.zeros(len(tmax))
     
     # for j in range(0,len(t)):
     #    errRKF[j] = abs(y(t[j])-yappRKF[j])
     # for j in range(0, len(tmin)):
     #    errRK4_min[j] = abs(y(tmin[j])-yappRK4_min[j])
     # for j in range(0, len(tmax)):
     #    errRK4_max[j] = abs(y(tmax[j])-yappRK4_max[j])
        
     # plt.plot(t,yappRKF,label = 'Runge-Kutta-Fehlberg')
     # plt.plot(tmin,yappRK4_min,label = 'RK4, h=0.05')
     # plt.plot(tmax,yappRK4_max,label = 'RK4, h=0.5')
     # plt.xlabel('t')
     # plt.ylabel('approx')
     # plt.legend(loc = 'best')
     # plt.title("Runge Kutta Method Comparison")
     # plt.show()
     # plt.plot(t,errRKF,label = 'Runge-Kutta-Fehlberg')
     # plt.plot(tmin,errRK4_min,label = 'RK4, h=0.05')
     # plt.plot(tmax,errRK4_max,label = 'RK4, h=0.5')
     # plt.yscale('log')
     # plt.xlabel('t')
     # plt.ylabel('absolute err')
     # plt.legend(loc = 'best')
     # plt.title("Runge Kutta Method Errors")
     # plt.show()

     # Problem 5:
     y = lambda t: np.exp(-t)-np.exp(t)+2*np.exp(2*t)
     ya = np.array([2,2])
     a = 0
     b = 1
     h = 0.1
     N = int((b-a)/h)
     
     [yapp_euler,t] = euler_systems(a,b,h,ya)
     [yapp_RK4,t4] = RK4systems(a,b,h,ya)

     y_euler = np.zeros(len(t))
     y_RK4 = np.zeros(len(t4))
     for i,y in enumerate(yapp_euler):
        y_euler[i] = y[0]
     for i,y in enumerate(yapp_RK4):
        y_RK4[i] = y[0]
     
     # err_euler = np.zeros(len(t))
     # err_RK4 = np.zeros(len(t4))

     err_euler = np.abs(solution_2(t)-y_euler)
     err_RK4 = np.abs(solution_2(t4)-y_RK4)

     plt.plot(t,err_euler)
     plt.yscale('log')
     plt.xlabel('t')
     plt.ylabel('absolute err')
     plt.legend(loc = 'upper left')
     plt.title("Error in Euler's Method for Systems of DEs")
     plt.show()

     plt.plot(t,err_RK4)
     plt.yscale('log')
     plt.xlabel('t')
     plt.ylabel('absolute err')
     plt.legend(loc = 'upper left')
     plt.title("Error in RK4 for Systems of DEs")
     plt.show()

     plt.plot(t,err_euler, label = 'Euler')
     plt.plot(t,err_RK4, label = 'RK4')
     plt.yscale('log')
     plt.xlabel('t')
     plt.ylabel('absolute err')
     plt.legend(loc = 'upper left')
     plt.title("Error Comparison")
     plt.show()


     return 
     

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


def RKFehlberg(a,b,hmin,hmax,ya,tol):
     h = hmax
     t = []
     yapp = []
     yapp.append(ya)
     t.append(a)
     flag = True
     step = 0
     while flag:
          k1 = h*eval_f(t[-1],yapp[-1])
          k2= h*eval_f(t[-1]+h/4, yapp[-1] + (1/4)*k1)
          k3 = h*eval_f(t[-1]+((3/8)*h), yapp[-1] + (3/32)*k1 + (9/32)*k2)
          k4 = h*eval_f(t[-1]+((12/13)*h), yapp[-1] + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
          k5 = h*eval_f(t[-1]+h, yapp[-1] + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
          k6 = h*eval_f(t[-1]+h/2, yapp[-1] - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)

          R= (1/h)*np.abs((1/360)*k1-(128/4275)*k3-(2197/75240)*k4+(1/50)*k5+(2/55)*k6)
          if R <=tol:
               t.append(t[-1] + h)
               yapp.append(yapp[-1] + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5)
               step +=1
          else:
               delta = 0.84*((tol/R)**0.25)
               if delta <= 0.1:
                    h = 0.1 * h
               elif delta >= 4:
                    h = 4*h
               else:
                    h = delta*h

          if h > hmax:
               h = hmax
          if t[-1] >= b:
               flag = False
          elif t[-1]+h > b:
               h = b - t[-1]
          elif h < hmin:
               print('min h exceeded. procedure was unsuccessful')
               flag = False

     return (np.array(yapp),np.array(t), step)


def euler_systems(a,b,h,ya):
    N = int((b-a)/h)
    yapp = np.zeros((N+1,2))
    t = np.zeros(N+1)
    yapp[0] = ya
    t[0] = a
    for jj in range(1, N+1):
        tj = a+(jj-1)*h
        t[jj] = tj+h
        yapp[jj] = euler_step(t[jj-1],yapp[jj-1],eval_f_sys,h)

    return (yapp,t)


def RK4systems(a,b,h,ya):
     N = int((b-a)/h)
     yapp = np.zeros((N+1,2))
     t = np.zeros(N+1)
     yapp[0] = ya
     t[0] = a
     for jj in range(1, N+1):
        tj = a+(jj-1)*h
        t[jj] = tj+h
        rk = yapp[jj-1]
        k1 = h*eval_f_sys(tj,rk)
        k2= h*eval_f_sys(tj+h/2,rk+0.5*k1)
        k3 = h*eval_f_sys(tj+h/2,rk+1/2*k2)
        k4 = h*eval_f_sys(tj+h,rk+k3)
        yapp[jj] = rk + 1/6*(k1+2*k2+2*k3+k4)

     return (yapp,t)

def eval_f(t,y):

     f = -(y+1)*(y+3)

     return f

def eval_f_sys(t,y):

     y1=y[0]
     y2=y[1]
     u1_p=y2
     u2_p=6*np.exp(-t)-2*y1+3*y2

     return np.array([u1_p,u2_p])

def euler_step(t,y,f,h):
    
    y_out=np.add(y,h*f(t,y))

    return y_out

def solution_2(t):
    return np.exp(-t)-np.exp(t)+2*np.exp(2*t)
     
driver()                    
     
