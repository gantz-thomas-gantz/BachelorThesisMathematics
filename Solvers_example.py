# Import the numpy library.
import numpy as np

# Import the Regression Problem classes from the Solvers_Interface module.
from Solvers.Solvers_Interface import RegressionProblem_LeastSquare
from Solvers.Solvers_Interface import RegressionProblem_Huber

# Initialise a point cloud with evenly spaced points along the x- and y-axes.
X_points = np.linspace(0, 10, 11)
Y_points = np.linspace(0, 10, 11)

# Add an outlier to Y_points to demonstrate robust regression.
Y_points[10] = 1

# Initialise a Regression Problem to be solved with Least Square method.
RP_LQ = RegressionProblem_LeastSquare(X_points, Y_points)

# Initialise a Regression Problem to be solved with Huber method.
RP_Hu = RegressionProblem_Huber(X_points, Y_points)

# Solve the Regression Problems with their respective methods.
# RP_LQ.solve(degree of the regression polynomial, maximum number of iterations of qpOASES)
poly_LQ, poly_LQ_coefficients = RP_LQ.solve(1, 100) 

# RP_Hu.solve(degree of the regression polynomial, maximum number of iterations of qpOASES, gamma)
poly_Hu, poly_Hu_coefficients = RP_Hu.solve(1, 100, 1) 

# Plot the regression curves for both Regression Problems.
RP_LQ.plot()
RP_Hu.plot()

# Show and save the plot as an image file with the specified name.
RP_LQ.show_and_save("some_nice_regression_polynomials")  # or RP_Hu.show_and_save()
