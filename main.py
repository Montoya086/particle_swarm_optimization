import numpy as np
import matplotlib.pyplot as plt


class PSO:
    """
    Particle Swarm Optimization algorithm
    Parameters:
    N: int
        Number of particles
    w: float
        Inertia weight
    c1: float
        Cognitive weight
    c2: float
        Social weight
    iterations: int
        Number of iterations
    threshold: float
        Threshold to stop the algorithm
    f: function
        Function to minimize
    """
    def __init__(self, N, w, c1, c2, iterations, threshold, f):
        self.f = f
        self.N = N
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.iterations = iterations
        self.threshold = threshold
        np.random.seed(42)
        self.x = np.random.uniform(-10, 10, (N, 2)) 
        self.v = np.random.uniform(-1, 1, (N, 2)) 
        self.pbest = self.x.copy() 
        self.gbest = self.x[np.argmin(f(self.x[:, 0], self.x[:, 1]))] 
        self.evolution = [self.x.copy()]

    
    def iterate(self):
        """
        Run the PSO algorithm
        """
        for _ in range(self.iterations):
            # Update the position and velocity of the particles
            r1, r2 = np.random.rand(2) 
            # Update the position and velocity of the particles
            self.v = self.w * self.v + self.c1 * r1 * (self.pbest - self.x) + self.c2 * r2 * (self.gbest - self.x)
            # Update the position of the particles
            self.x += self.v
            # Update the personal best and global best
            self.fitness = self.f(self.x[:, 0], self.x[:, 1])
            better_mask = self.fitness < self.f(self.pbest[:, 0], self.pbest[:, 1])
            self.pbest[better_mask] = self.x[better_mask]
            # Update the global best
            if np.min(self.fitness) < self.f(self.gbest[0], self.gbest[1]):
                self.gbest = self.x[np.argmin(self.fitness)]
            
            # Save the evolution
            self.evolution.append(self.x.copy())
            # Check the threshold to stop the algorithm
            if self.f(self.gbest[0], self.gbest[1]) < self.threshold:
                break

    def get_best_solution(self):
        """
        Get the best solution found
        """
        return self.gbest
    
    def plot_evolution(self, num_plots=3):
        """
        Plot the evolution of the particles
        Parameters:
        num_plots: int
            Number of plots to show (default 3)
        """
        # If num_plots is greater than the number of iterations, set it to the number of iterations
        if num_plots > len(self.evolution):
            num_plots = len(self.evolution)
        # Get the indices to plot
        indices = np.linspace(0, len(self.evolution) - 1, num=num_plots, dtype=int)
        plt.figure(figsize=(5 * num_plots, 5))
        # Plot each iteration
        for j, idx in enumerate(indices):
            points = self.evolution[idx]
            plt.subplot(1, num_plots, j + 1)
            plt.contourf(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100),
                        self.f(*np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))))
            plt.scatter(points[:, 0], points[:, 1], color='r')
            plt.scatter(self.gbest[0], self.gbest[1], color='g')
            plt.annotate(f'({self.gbest[0]:.2f}, {self.gbest[1]:.2f})', (self.gbest[0], self.gbest[1]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.title(f'Iteration {idx}')
        plt.show()


if __name__ == '__main__':
    # Function to minimize
    def f(x, y):
        return (x - 3)**2 + (y - 2)**2

    # Parameters
    N = 40 
    w = 0.5 
    c1 = c2 = 1.5
    iterations = 100
    threshold = 1e-6

    # Initialize PSO
    pso = PSO(N, w, c1, c2, iterations, threshold, f)
    # Run PSO
    pso.iterate()

    # Get best solution
    print('Minimum found at:', pso.get_best_solution())
    # Plot evolution
    pso.plot_evolution(3)

