# This file includes two solvers using the Least Squares method via qpOASES.
# The first one finds a most approximate solution for AX=B.
# The second one finds an approximative polynomial function for a given dataset.

# Import.
import numpy as np
from qpoases import PyQProblemB as QProblemB
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel

def Solver_LeastSquare_LSE(A, B, max_number_of_iterations):
    
    # Setup data of QP.
    
    H   = 2*np.transpose(A)@A
    nV = len(H)
    
    g   = (-2*B@A).ravel()
    lb  = np.full(nV,-np.inf) 
    ub  = np.full(nV,np.inf)

    # Initializing QProblem object.
    
    QP = QProblemB(nV)

    # Setting and printing Options.

    options = Options()
    options.printLevel = PrintLevel.NONE
    QP.setOptions(options)

    # Setting up and solving QProblem object.

    nWSR = np.array([max_number_of_iterations]) # maximum iterations
    QP.init(H, g, lb, ub, nWSR)

    # Returning solution.

    xOpt = np.zeros(nV)
    QP.getPrimalSolution(xOpt)
    return xOpt

def Solver_LeastSquare_PolynomialRegression(X,Y,n,max_number_of_iterations):

    B = Y
    A = np.empty((len(X),n+1))
    for i in range(0,len(X)):
        X_row = np.empty(n+1)
        for j in range(0,n+1):
            X_row[j] = X[i]**j
        A[i] = X_row

    PolynomialCoefficients = Solver_LeastSquare_LSE(A,B,max_number_of_iterations)

    def RegressionPolynomial(x):
        value = 0
        for i in range(0,len(PolynomialCoefficients)):
            value += PolynomialCoefficients[i]*x**i
        return value
    
    return RegressionPolynomial, PolynomialCoefficients

    








