import numpy as np
import random
import matplotlib.pyplot as plt
from Solvers.Solvers_Interface import RegressionProblem_LeastSquare
from Solvers.Solvers_Interface import RegressionProblem_Huber

number_of_iterations = 1000000

# Maximum degree of regression polynomial.
for n in [1,2,3]:

    # Points along a polynomial with a different amount of outliers (1st experiment).
    # Random polynomial of given maximum degree.
    coefficients = []
    for i in range(n+1):
        coefficients.append(random.random())

    def Polynomial(x,coefficients):
        y = 0
        for i in range(len(coefficients)):
            y += coefficients[i]*x**i
        return y

    # Different levels of noise.
    for noise_percentage in [0.01,0.03,0.1]:
        # Making the data points.
        X_points = np.linspace(0, 5, 11)
        X = np.linspace(np.min(X_points), np.max(X_points), 1000)
        Y_points_initial = np.array(Polynomial(X_points, coefficients))
        
        # Adding noise.
        noise = np.random.normal(0, noise_percentage, len(Y_points_initial))
        Y_points_noisy = Y_points_initial + noise

        # Different amount of outliers and plotting the points.
        for number_of_outliers in [0,1,2]:
            fig = plt.figure(figsize=(10, 6))
            plt.rc('text', usetex=True)
            Y_points = Y_points_initial.copy()
            
            if (number_of_outliers==0):

                # "Removing" the outlier for one data set.
                X_points_without_outliers = X_points
                Y_points_without_outliers = Y_points
            
            if (number_of_outliers==1):
                
                # Adding an outlier.
                random_index_1 = random.randint(0, len(X_points)-1)
                Y_points[random_index_1] *= 3
                
                # Removing the outlier for one data set.
                X_points_without_outliers = np.delete(X_points, random_index_1)
                Y_points_without_outliers = np.delete(Y_points, random_index_1)
           
            if (number_of_outliers==2):
                
                # Adding 2 outliers
                random_index_1 = random.randint(0, len(X_points)-1)
                Y_points[random_index_1] *= 3
                random_index_2 = random.randint(0, len(X_points)-1)
                while random_index_2 == random_index_1:
                    random_index_2 = random.randint(0, len(X_points)-1)
                Y_points[random_index_2] *= 3

                # Removing the outliers for one data set.
                X_points_without_outliers = np.delete(X_points, [random_index_1, random_index_2])
                Y_points_without_outliers = np.delete(Y_points, [random_index_1, random_index_2])
                
            # Making the table.
            data = []

            # Solving with Least Squares and removed outliers.
            data_temp = np.zeros(n+1)
            for i in range(number_of_iterations):
                RP = RegressionProblem_LeastSquare(X_points_without_outliers,Y_points_without_outliers)
                RegressionPolynomial, RegressionCoefficients = RP.solve(n,1000)
                data_temp += np.abs(coefficients/RegressionCoefficients)

            data_temp = data_temp/number_of_iterations
            data.append(["Least Squares (rem. out.)", [round(x, 2) for x in data_temp]])

            # Solving with Huber and different gammas.
            for gamma in [0.5, 1.0, 2.0]:
                
                data_temp = np.zeros(n+1)
                for i in range(number_of_iterations):

                    RP = RegressionProblem_Huber(X_points,Y_points)
                    RegressionPolynomial, RegressionCoefficients = RP.solve(n,1000,gamma)
                    data_temp += np.abs(coefficients/RegressionCoefficients)
                    
                data_temp = data_temp/number_of_iterations
                data.append([r'Huber ($\gamma$ = %1.1f)' % gamma, [round(x, 2) for x in data_temp]])

            # Solving with Least Squares.
            data_temp = np.zeros(n+1)
            for i in range(number_of_iterations):
                
                RP = RegressionProblem_LeastSquare(X_points,Y_points)
                RegressionPolynomial, RegressionCoefficients = RP.solve(n,1000)
                data_temp += np.abs(coefficients/RegressionCoefficients)

            data_temp = data_temp/number_of_iterations
            data.append(["Least Squares (stand.)", [round(x, 2) for x in data_temp]])
            
            # Setting options for the table.
            table = plt.table(cellText=data, colLabels=["Method", "InitialCoefficients/NumericalCoefficients"], loc='center')
            table.scale(2, 3.86)
            table.auto_set_font_size(False)
            table.set_fontsize(22)
            table.auto_set_column_width([0, 1])
            plt.axis('off')
            
            # Setting options for the figure and saving it.
            plt.tight_layout()
            plt.savefig("InitialCoefficients_ratio_NumericalCoefficients_n=%i,outliers:%s,noise_percentage=%f" % (n, number_of_outliers, noise_percentage) + ".png", bbox_inches='tight')
            plt.close(fig)

    


