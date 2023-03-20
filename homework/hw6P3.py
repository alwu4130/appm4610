import numpy as np
import math
import matplotlib.pyplot as plt


def driver():

# this demo code considers the boundary value problem
# y'' = p(x)y'+q(x)y + r(x)
#y(a) = alpha, y(b) = beta

# boundary data
     a = 0
     b = 1
     alpha = np.pi
     beta = -np.pi

# step size
     hvals = [0.1, 0.05, 0.025]
     # N = int((b-a)/h)
     
     # x = np.linspace(a,b,N+1)
     
     # yapp = make_FDmatDir(x,h,N,alpha,beta)
          
#  exact solution 
     # c2 = 1/70*(8-12*np.sin(np.log(2))-4*np.cos(np.log(2)))
     # c1 = 11/10-c2
     # y = lambda x: c1*x+c2/(x**2)-3/10*np.sin(np.log(x))-1/10*np.cos(np.log(x))
     y = lambda x: np.sin(np.pi * x) 

     for h in hvals:
          N = int((b-a)/h) 
          x = np.linspace(a,b,N+1)
          yex = y(x)
          yapp = make_FDmatDir(x,h,N,alpha,beta)
          err = np.abs(yex-yapp)
          ylabel = str("h = "+ str(h))
          plt.semilogy(x, err, label = ylabel)
          plt.legend(loc = 'best')
          plt.xlabel("x")
          plt.ylabel("Error")
          plt.title("Absolute Error of BVP Using Centered Difference")
     # print(condA)     
     plt.show()       
     # yex = y(x)
    
     # plt.plot(x,yapp,label = 'FD aprox')
     # plt.plot(x,yex,label = 'Exact')
     # plt.xlabel('x')
     # plt.legend(loc = 'upper left')
     # plt.show()
     
     # err = np.zeros(N+1)
     # for j in range(0,N+1):
     #      err[j] = abs(yapp[j]-yex[j])
          
     # print('err = ', err)
          
     # plt.plot(x,err,label = 'FD aprox')
     # plt.xlabel('x')
     # plt.xlabel('absolute error')
     # plt.legend(loc = 'upper left')
     # plt.show()
          
     return
     
def eval_pqr(x):

     p = np.zeros(len(x))
     q = 2*np.ones(len(x))
     r = -((np.pi**2)*(np.sin(np.pi*x))) - (2*np.sin(np.pi*x))   

     return(p,q,r)     



def make_FDmatDir(x,h,N,alpha,beta):

# evaluate coefficients of differential equation
     (p,q,r) = eval_pqr(x)
     
 
# create the finite difference matrix     
     Matypp = 1/h**2*(np.diag(2*np.ones(N+1)) -np.diag(np.ones(N),-1) - 
           np.diag(np.ones(N),1))


     # Matyp = 1/(2*h)*(np.diag(np.ones(N-2),1)-np.diag(np.ones(N-2),-1))
     
     A = Matypp + np.diag(q)
     A[0][:] = np.zeros(N+1)
     # A[0][0] = 1/h
     A[0][1] = 1/(2*h)
     A[N][:] = np.zeros(N+1)
     # A[N][N] = -1/h
     A[N][N-1] = -1/(2*h)

# create the right hand side rhs: (N-1) in size
     rhs = -r
#  update with boundary data   
     # rhs[0] = rhs[0] + (1/h**2-1/(2*h)*-p[1])*alpha
     # rhs[N-2] = rhs[N-2] + (1/h**2+1/(2*h)*-p[N-1])*beta
     rhs[0] = alpha
     rhs[-1] = beta
     
# solve for the approximate solution

     # Ainv = np.linalg.inv(A)
     # sol = np.matmul(Ainv,rhs)
     sol =np.linalg.solve(A, rhs)
     
     # yapp = np.zeros(N+1)
     # yapp[0] = alpha
     # for j in range(1,N):
     #     yapp[j] = sol[j-1]
     # yapp[N] = beta    

     return sol
     
driver()     
