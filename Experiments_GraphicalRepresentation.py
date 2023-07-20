import numpy as np
import random
import matplotlib.pyplot as plt
from Solvers.Solvers_Interface import RegressionProblem_LeastSquare
from Solvers.Solvers_Interface import RegressionProblem_Huber

# Maximum degree of regression polynomial.
for n in [1, 2, 3]:
    
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
    for noise_percentage in [0.01, 0.03, 0.1]:
        
        # Making the data points.
        X_points = np.linspace(0, 5, 11)
        X = np.linspace(np.min(X_points), np.max(X_points), 1000)
        Y_points_initial = np.array(Polynomial(X_points, coefficients))
        
        # Adding noise.
        noise = np.random.normal(0, noise_percentage, len(Y_points_initial))
        Y_points_noisy = Y_points_initial + noise

        # Different amount of outliers and plotting the points.
        for number_of_outliers in [0,1,2]:
            fig = plt.figure(figsize=(6, 6))
            ax1 = fig.add_subplot(111)
            fig_table = plt.figure(figsize=(15, 5))
            ax2 = fig_table.add_subplot(111)
            plt.rc('text', usetex=True)
            Y_points = Y_points_initial.copy()
            
            if (number_of_outliers==0):

                # "Removing" the outlier for one data set.
                X_points_without_outliers = X_points
                Y_points_without_outliers = Y_points
            
                # Plotting the points.
                ax1.plot(X_points, Y_points, marker='o', markersize=5, color='gray', linestyle='None', label="points")
            
            if (number_of_outliers==1):
                
                # Adding an outlier.
                random_index_1 = random.randint(0, len(X_points)-1)
                Y_points[random_index_1] *= 3
                
                # Removing the outlier for one data set.
                X_points_without_outliers = np.delete(X_points, random_index_1)
                Y_points_without_outliers = np.delete(Y_points, random_index_1)

                # Plotting the points.
                ax1.plot(X_points, Y_points, marker='o', markersize=5, color='gray', linestyle='None', label="points")
                ax1.plot([X_points[random_index_1]], [Y_points[random_index_1]], marker='o', markersize=10, color='magenta', linestyle='None', label="outliers")
           
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

                # Plotting the points
                ax1.plot(X_points, Y_points, marker='o', markersize=5, color='gray', linestyle='None', label="points")
                ax1.plot([X_points[random_index_1], X_points[random_index_2]], [Y_points[random_index_1], Y_points[random_index_2]], marker='o', markersize=10, color='magenta', linestyle='None', label="outliers")
                
            # Making the table.
            data = []
            data.append(["Initial", [round(x, 2) for x in coefficients]])

            # Solving with Least Squares and removed outliers.
            RP = RegressionProblem_LeastSquare(X_points_without_outliers,Y_points_without_outliers)
            RegressionPolynomial, RegressionCoefficients = RP.solve(n,1000)

            data.append(["Least Squares (rem. out.)", [round(x, 2) for x in RegressionCoefficients]])
            Y = RegressionPolynomial(X)
            ax1.plot(X,Y, label="Least Squares (rem. out.)")

            # Solving with Huber and different gammas.
            for gamma in [2,1,0.5]:

                RP = RegressionProblem_Huber(X_points,Y_points)
                RegressionPolynomial, RegressionCoefficients = RP.solve(n,1000,gamma)
                
                data.append([r'Huber ($\gamma$ = %1.1f)' % gamma, [round(x, 2) for x in RegressionCoefficients]])

                Y = RegressionPolynomial(X)
                ax1.plot(X, Y, label=r'Huber ($\gamma$ = %1.1f)' % gamma)

            # Solving with Least Squares.
            RP = RegressionProblem_LeastSquare(X_points,Y_points)
            RegressionPolynomial, RegressionCoefficients = RP.solve(n,1000)

            data.append(["Least Squares (stand.)", [round(x, 2) for x in RegressionCoefficients]])
            Y = RegressionPolynomial(X)
            ax1.plot(X,Y, label="Least Squares (stand.)")

            # Setting options for the table.
            ax2.axis('off')
            table = ax2.table(cellText=data, colLabels=['Label', 'Coefficients'], loc='center')
            plt.tight_layout()

            # Setting options for the graph.
            ax1.set_xlabel('x', fontsize=25)
            ax1.set_ylabel('RegressionPolynomial(x)', fontsize=25)
            ax1.tick_params(axis='x', labelsize=20)   
            ax1.tick_params(axis='y', labelsize=20)
            ax1.yaxis.set_label_coords(-0.135, 0.5)
            ax1.grid(True)
            
            # Setting options for the figure and saving it.
            plt.tight_layout()
            table_filename = "n=%i,outliers:%s,noise_percentage=%f" % (n, number_of_outliers, noise_percentage) + "_table.png"
            graph_filename = "n=%i,outliers:%s,noise_percentage=%f" % (n, number_of_outliers, noise_percentage) + "_graph.png"
            fig.savefig(graph_filename, bbox_inches='tight')
            fig_table.savefig(table_filename, bbox_inches='tight')

            # Closing the figures to free memory.
            plt.close(fig)
            plt.close(fig_table)
    
    # Making random points and plotting them (2nd experiment)
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    fig_table = plt.figure(figsize=(15, 5))
    ax2 = fig_table.add_subplot(111)
    plt.rc('text', usetex=True)

    for i in range(len(X_points)):
        Y_points[i] = random.uniform(0, 5)
            
    ax1.plot(X_points, Y_points, marker='o', markersize=5, color='gray', linestyle='None', label="points")
            
    # Making the table.
    data = []

    # Solving with Huber and different gammas.
    for gamma in [0.5, 1.0, 2.0]:

        RP = RegressionProblem_Huber(X_points,Y_points)
        RegressionPolynomial, RegressionCoefficients = RP.solve(n,1000,gamma)
                
        data.append([r'Huber ($\gamma$ = %1.1f)' % gamma, [round(x, 2) for x in RegressionCoefficients]])

        Y = RegressionPolynomial(X)
        ax1.plot(X, Y, label=r'Huber ($\gamma$ = %1.1f)' % gamma)

    # Solving with Least Squares.
    RP = RegressionProblem_LeastSquare(X_points,Y_points)
    RegressionPolynomial, RegressionCoefficients = RP.solve(n,1000)

    data.append(["Least Squares (stand.)", [round(x, 2) for x in RegressionCoefficients]])
    Y = RegressionPolynomial(X)
    ax1.plot(X,Y, label="Least Squares (stand.)")

    # Setting options for the table.
    ax2.axis('off')
    table = ax2.table(cellText=data, colLabels=['Label', 'Coefficients'], loc='center')
    plt.tight_layout()

    # Setting options for the graph.
    ax1.set_xlabel('x', fontsize=25)
    ax1.set_ylabel('RegressionPolynomial(x)', fontsize=25)
    ax1.tick_params(axis='x', labelsize=20)   
    ax1.tick_params(axis='y', labelsize=20)
    ax1.yaxis.set_label_coords(-0.135, 0.5)
    ax1.grid(True)
            
    # Setting options for the figure and saving it.
    plt.tight_layout()
    table_filename = "random,n=%i" % n + "_table.png"
    graph_filename = "random,n=%i" % n + "_graph.png"
    fig.savefig(graph_filename, bbox_inches='tight')
    fig_table.savefig(table_filename, bbox_inches='tight')

    # Closing the figures to free memory.
    plt.close(fig)
    plt.close(fig_table)




