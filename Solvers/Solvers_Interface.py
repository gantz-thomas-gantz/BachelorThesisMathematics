from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from Solvers.Solvers_LeastSquare import Solver_LeastSquare_PolynomialRegression
from Solvers.Solvers_Huber import Solver_Huber_PolynomialRegression

class RegressionProblem:
    __metaclass__ = ABCMeta

    # member variables.
    X_points = []
    Y_points = []
    ProblemClass = ""

    used_n = -1
    used_max_number_of_iterations = -1

    def f(x):
        return 0
    RegressionPolynomial = f

    PolynmialCoefficients = []

    # constructor.
    def __init__(self, X_points, Y_points):
        self.X_points = X_points
        self.Y_points = Y_points

    # member methods.
    @abstractmethod
    def solve(self,max_number_of_iterations):
        pass

    @abstractmethod
    def plot(self):
        pass

    def show(self):
        plt.plot(self.X_points, self.Y_points, 'ro', label="points")

        # naming the x axis.
        plt.xlabel('x')
        # naming the y axis.
        plt.ylabel('RegressionPolynomial(x)')
    
        # giving a title to my graph.
        plt.title("maximum number of iterations = %i" % self.used_max_number_of_iterations)
        # showing the legend.
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), fancybox=True, shadow=True, ncol=1, fontsize=13)

        # adding grid.
        plt.grid(True)
    
        # function to show the plot.
        plt.show()

    def show_and_save(self,picturename):
        plt.plot(self.X_points, self.Y_points, 'ro', label="points")

        # naming the x axis.
        plt.xlabel('x')
        # naming the y axis.
        plt.ylabel('RegressionPolynomial(x)')
    
        # giving a title to my graph.
        plt.title("maximum number of iterations = %i" % self.used_max_number_of_iterations)

        # showing the legend.
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), fancybox=True, shadow=True, ncol=1, fontsize=13)
        plt.savefig(picturename+".png", bbox_inches='tight')

        # adding grid.
        plt.grid(True)

        # function to show the plot.
        plt.show()


class RegressionProblem_LeastSquare(RegressionProblem):
    
    # member methods.
    def solve(self,n,max_number_of_iterations):
        self.used_n = n
        self.used_max_number_of_iterations = max_number_of_iterations
        self.RegressionPolynomial, self.PolynmialCoefficients = Solver_LeastSquare_PolynomialRegression(self.X_points,self.Y_points,n,max_number_of_iterations)
        return self.RegressionPolynomial, self.PolynmialCoefficients 
    
    def plot(self):

        # x axis values.
        X = np.linspace(np.min(self.X_points),np.max(self.Y_points),1000)
        # corresponding y axis values.
        Y = self.RegressionPolynomial(X)
    
        # plotting the regression polynomial. 
        plt.plot(X,Y, label="Regression Polynomial (Least Square) of degree %i" % self.used_n)

class RegressionProblem_Huber(RegressionProblem):
    
    #member variables.
    used_gamma = -1.
    
    #member methods.
    def solve(self,n,max_number_of_iterations,gamma):
        self.used_n = n
        self.used_max_number_of_iterations = max_number_of_iterations
        self.used_gamma = gamma
        self.RegressionPolynomial, self.PolynmialCoefficients = Solver_Huber_PolynomialRegression(self.X_points,self.Y_points,n,max_number_of_iterations,gamma)
        return self.RegressionPolynomial, self.PolynmialCoefficients
    
    def plot(self):

        # x axis values.
        X = np.linspace(np.min(self.X_points),np.max(self.Y_points),1000)
        # corresponding y axis values.
        Y = self.RegressionPolynomial(X)
    
        # plotting the regression polynomial. 
        plt.rc('text', usetex=True)
        plt.plot(X,Y, label=r'Regression Polynomial (Huber) of degree %i with $\gamma$ = %1.1f' % (self.used_n,self.used_gamma))
    
