import numpy as np
import matplotlib.pyplot as plt


class WindFarm:
    def __init__(self, n_turbines, locations):
        self.n_turbines = n_turbines
        self.locations = locations
        # Center constraint
        self.radius1 = 2
        self.x1, self.y1 = (5,5)
        # Bottom left constraint
        self.radius2 = 1
        self.x2, self.y2 = (0,0)
        # Top left constraint
        self.x3, self.y3 = (0,10)

    def efficiency(self):
        distances = np.zeros((self.n_turbines, self.n_turbines))
        # TODO: theoretically only need to calculate half of the distances
        for i in range(self.n_turbines):
            for j in range(self.n_turbines):
                distances[i, j] = np.linalg.norm(self.locations[i] - self.locations[j])
        
        efficiency = 1 / (1 + (1/distances))
        return np.sum(efficiency) / 2 # To avoid double counting
    
    def constraints(self):
        # Just gonna start with a basic count constraint
        c = np.zeros(3)
        for i in range(self.n_turbines):
            if np.linalg.norm(self.locations[i] - (self.x1, self.y1)) <= self.radius1:
                c[0] += 1
            if np.linalg.norm(self.locations[i] - (self.x2, self.y2)) <= self.radius2:
                c[1] += 1
            if np.linalg.norm(self.locations[i] - (self.x3, self.y3)) <= self.radius2:
                c[2] += 1
        return c
    
    def plot(self):
        # Plot the turbines
        plt.scatter(self.locations[:,0], self.locations[:,1])
        # Plot the constraints
        circle1 = plt.Circle((self.x1, self.y1), self.radius1, fill=False)
        circle2 = plt.Circle((self.x2, self.y2), self.radius2, fill=False)
        circle3 = plt.Circle((self.x3, self.y3), self.radius2, fill=False)
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.gca().add_patch(circle3)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.axis('equal')  # Make the plot aspect ratio 1:1
        plt.show()

        

        
    
    

