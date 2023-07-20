# This file includes two solvers using the Huber cost function via qpOASES.
# The first one finds a most approximate solution for AX=B.
# The second one finds an approximative polynomial function for a given dataset.

# Import.
import numpy as np
from qpoases import PyQProblem as QProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel

def Solver_Huber_LSE(A,B,max_number_of_iterations,gamma):

    d = len(A[1]) # d = X_size 
    l = len(A)    # l = Z_size = T_size = B_size
    
    n = d + l + l
    m = l + l

    # Setup data of QP.
    
    H   = np.zeros((n,n))
    for i in range(d,d+l):
        H[i][i] = 1
    nV = len(H) # nV = n

    A_QP = np.concatenate((np.concatenate((A,-1*A),axis=0),np.concatenate((-1*np.identity(l),np.identity(l)),axis=0),np.concatenate((-1*np.identity(l),-1*np.identity(l)),axis=0)),axis=1) 
    nC = len(A_QP) # nC = l + l

    g = np.zeros(n)
    for i in range(d+l,n):
        g[i] = gamma
    
    lb = np.full(nV,-np.inf) 
    ub = np.full(nV,np.inf)

    lbA_QP = np.full(nV,-np.inf)
    ubA_QP = np.concatenate((B,-1*B),axis=None)

    # Initializing QProblem object.
    
    QP = QProblem(nV,nC)

    # Setting and printing Options.

    options = Options()
    options.printLevel = PrintLevel.NONE 
    QP.setOptions(options)

    # Setting up and solving QProblem object.

    nWSR = np.array([max_number_of_iterations]) # maximum iterations
    QP.init(H, g, A_QP, lb, ub, lbA_QP, ubA_QP, nWSR)

    # Obtaining solution of the QP.

    xOpt = np.zeros(nV)
    QP.getPrimalSolution(xOpt)
    
    # Returning actual searched X.
    X_solution = xOpt[0:d]
    return X_solution 
    

def Solver_Huber_PolynomialRegression(X,Y,n,max_number_of_iterations,gamma):

    B = Y
    A = np.empty((len(X),n+1))
    for i in range(0,len(X)):
        X_row = np.empty(n+1)
        for j in range(0,n+1):
            X_row[j] = X[i]**j
        A[i] = X_row

    PolynomialCoefficients = Solver_Huber_LSE(A,B,max_number_of_iterations,gamma)

    def RegressionPolynomial(x):
        value = 0
        for i in range(0,len(PolynomialCoefficients)):
            value += PolynomialCoefficients[i]*x**i
        return value
    
    return RegressionPolynomial, PolynomialCoefficients