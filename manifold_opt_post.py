"""
Created on Sun Dec  9 17:22:41 2018

@author: ap0


We generate a random matrices A of size M x N and a random s-row sparse
matrix X of size K x N.

1. First rewrite Y=YT * Y0 
2. Solve AW=YT
3. Recover X=YT * Y0

"""

from autograd import numpy as np
from   autograd.numpy import linalg as LA
import matplotlib.pyplot as plt

from pymanopt.manifolds import FixedRankEmbedded
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient


# Dimensions
N = 100
M = 15
s = 5
K = 7
r = s
maxiter = 1000

# Hyperparameters
delta = 0.03
lambd = 8


# The uknown matrix with row-sparisity s
X0 = np.random.normal(0,1,[N,K])
arr = np.arange(N)
np.random.shuffle(arr)
supp_comp = arr[0:N-s]
for ind in supp_comp:
    X0[ind,:] = 0  

# The measurement matrix (normalized)
A = np.random.normal(0,1,[M,N])
A = np.matmul(A,LA.inv(np.diag(LA.norm(A, axis=0))))
    
# The data matrix
Y0 = np.matmul(A,X0)
uu,vv,dd = LA.svd(Y0)
UY = dd[0:r,:]
YT = np.dot(uu[:,0:r],np.diag(vv[0:r]))



# Solving the manifold optiization problem
def fixedrank(A,YT,r):    
    """ Solves the AX=YT problem on the manifold of r-rank matrices with  
    """
    
    # Instantiate a manifold
    manifold = FixedRankEmbedded(N, r, r)
    
    # Define the cost function (here using autograd.numpy)
    def cost(X):
        U = X[0]
        cst = 0
        for n in range(N):
            cst = cst+huber(U[n,:])
        Mat = np.matmul(np.matmul(X[0],np.diag(X[1])),X[2])
        fidelity = LA.norm(np.subtract(np.matmul(A,Mat),YT))      
        return cst + lambd * fidelity**2    
    
    problem = Problem(manifold = manifold, cost = cost)
    solver = ConjugateGradient(maxiter = maxiter)    
    
    # Let Pymanopt do the rest
    Xopt = solver.solve(problem)
    
    #Solve
    Sol = np.dot(np.dot(Xopt[0],np.diag(Xopt[1])),Xopt[2])
    
    return Sol  

# Compute the huber function
def huber(u):
    if LA.norm(u) < delta:
        val = LA.norm(u)**2/(2 * delta)
    else:
        val = LA.norm(u) - delta/2
    return val  

# Recover the X = Sol_man
W = fixedrank(A,YT,r)
Sol_man = np.dot(W,UY)


# Error
err = LA.norm(np.subtract(X0,Sol_man)) * 1/(np.sqrt(K*N))
        
     
# Plotting
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 10))
textstr = '%d vectors of size %d\n Each measured %d times' %(K,N,M)
plt.axis('off')
plt.suptitle('Jointly sparse reconstruction of \n'+ textstr+'\n', fontsize = 15)
cols = ['Original frames\n', 'Non-convex\n']
axes[0].imshow(X0)
axes[0].axis('off')
axes[1].imshow(Sol_man)
axes[1].axis('off')
for ax, col in zip(axes, cols):
    ax.set_title(col)
plt.show()



form = np.vectorize(lambda f: format(f, '0.1e'))
print('ManOpt error =\n', form(err))




