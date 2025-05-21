import numpy as np
import matplotlib.pyplot as plt


class WindFarm:
    def __init__(self, turbines):
        self.turbines = turbines
        self.n_turbines = len(turbines)

        # Turbine constraints
        self.x1, self.y1 = (5,5)
        self.radius1 = 2
        self.x2, self.y2 = (0,0)
        self.radius2 = 1
        self.x3, self.y3 = (0,10)

        # Update the distances to all turbines
        # O(N^2) complexity
        for i in range(self.n_turbines):
            for j in range(self.n_turbines):
                self.turbines[i].update_distance(self.turbines[j])

    # Add the efficiency of all turbines
    def efficiency(self):
        total_efficiency = 0
        for turbine in self.turbines:
            total_efficiency += turbine.efficiency()
        return total_efficiency
    
    # Plot the turbines
    def plot(self):
        # Plot the turbines
        for turbine in self.turbines:
            plt.scatter(turbine.x, turbine.y)
        # Plot the constraints
        circle1 = plt.Circle((self.x1, self.y1), self.radius1, fill=False)
        circle2 = plt.Circle((self.x2, self.y2), self.radius2, fill=False)
        circle3 = plt.Circle((self.x3, self.y3), self.radius2, fill=False)
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.gca().add_patch(circle3)
        plt.axis('equal')  # Make the plot aspect ratio 1:1
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.show()
    
    # Close the plot
    def __del__(self):
        plt.close()

        
class Turbine:
    def __init__(self, location, n_turbines, i):
        self.location = location
        self.x = location[0]
        self.y = location[1]
        self.i = i
        self.distances = np.zeros(n_turbines)
        self.min_distance = np.inf
        self.min_distance_index = 0
    
    # Update the distance to a specific turbine
    def update_distance(self, turbine):
        # Update the distance to the turbine
        self.distances[turbine.i] = np.linalg.norm(self.location - turbine.location)

        # Update the minimum distance and index if the distance is smaller
        if self.distances[turbine.i] < self.min_distance:
            self.min_distance = self.distances[turbine.i]
            self.min_distance_index = turbine.i

    # Returns the distance to the closest turbine and its index
    def min_distance(self):
        return self.min_distance, self.min_distance_index
    
    # Returns the efficiency of the turbine
    def efficiency(self):
        return 1 / (1 + (1/self.min_distance))
    
        
    
    

