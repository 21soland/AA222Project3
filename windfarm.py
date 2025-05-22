import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-6

class WindFarm:
    def __init__(self, n_turbines):
        self.turbines = None
        self.n_turbines = n_turbines

        # Turbine constraints
        self.x1, self.y1 = (5,5)
        self.radius1 = 2
        self.x2, self.y2 = (0,0)
        self.radius2 = 1
        self.x3, self.y3 = (0,10)

        # Create and distribute the turbines
        self.redistribute()

        # Update the distances to all turbines
        self.update_all_distances()


    # Add the efficiency of all turbines
    def efficiency(self):
        total_efficiency = 0
        for turbine in self.turbines:
            total_efficiency += turbine.efficiency()
        return total_efficiency
    
    # Redistribute the turbines
    def redistribute(self):
        # Clear the turbines
        self.turbines = []

        # Uniformly distribute the turbines
        permutation = np.random.permutation(self.n_turbines)
        grid_size = 10
        scale_factor = grid_size / self.n_turbines
        for n in range(self.n_turbines):
            # Use a randomized latin square method
            x = round((np.random.random() + n) * scale_factor,2)
            y = round((np.random.random() + permutation[n]) * scale_factor,2)

            # If the point doesn't lie in the constraints, then just place it completely randomly until it does
            while not self.check_constraints(x,y):
                x = np.random.random() * grid_size
                y = np.random.random() * grid_size

            # Add the turbine to the list
            self.turbines.append(Turbine(np.array([x,y]), self.n_turbines, n))
    
    # Check if a point lies within the constraints
    def check_constraints(self, x,y):
        x1, y1 = 5,5
        r1 = 2
        x2, y2 = 0,0
        r2 = 1
        x3, y3 = 0,10
        r3 = 1
        if(x <= 0 or x >= 10 or y <= 0 or y >= 10):
            return False
        if (x-x1)**2 + (y-y1)**2 < r1**2:
            return False
        if (x-x2)**2 + (y-y2)**2 < r2**2:
            return False
        if (x-x3)**2 + (y-y3)**2 < r3**2:
            return False
        return True
    
    # Update the distances to all turbines
    def update_all_distances(self):
        # O(N^2) complexity
        for i in range(self.n_turbines):
            for j in range(self.n_turbines):
                if i != j:
                    self.turbines[i].update_distance(self.turbines[j], False)

    def set_turbines(self, turbines):
        self.turbines = turbines
        self.update_all_distances()
    
    
    # Cross-entropy optimize each turbine
    def cross_entropy_optimize(self, n_iterations, samples, cycles):
        # Cycle through the full turbine list
        n_best = 3
        sigma1 = 5
        sigma2 = sigma1
        print(f"Starting cross-entropy optimization with {n_iterations} iterations, {samples} samples, and {cycles} cycles")
        print(f"Initial cross-entropy efficiency: {self.efficiency()}")
        for cycle in range(cycles):
            print(f"Cycle {cycle} of {cycles}, Current efficiency: {self.efficiency()}")
            for i in range(self.n_turbines):
                # Get the n_best efficiencies, x, and y
                last_nbest_efficiencies = np.zeros(n_best)
                last_nbest_x = np.zeros(n_best)
                last_nbest_y = np.zeros(n_best)
                # Initialize all values to current efficiency

                current_efficiency = self.efficiency()
                last_nbest_efficiencies.fill(current_efficiency)
                last_nbest_x.fill(self.turbines[i].x)
                last_nbest_y.fill(self.turbines[i].y)

                new_nbest_efficiencies = last_nbest_efficiencies.copy()
                new_nbest_x = last_nbest_x.copy()
                new_nbest_y = last_nbest_y.copy()

                for q in range(n_iterations):
                    # Generate samples of random points from normal distribution centered on current turbine
                    x_samples = np.random.normal(0, sigma2, samples)
                    y_samples = np.random.normal(0, sigma2, samples)
                    # Try each sample point
                    for j in range(samples):
                        # Evenly sample the last n_best points
                        pointx = x_samples[j] + last_nbest_x[j % n_best]
                        pointy = y_samples[j] + last_nbest_y[j % n_best]
                        # Check if the point is within constraints
                        if self.check_constraints(pointx, pointy):
                            # Update the location of the turbine
                            self.turbines[i].update_location(pointx, pointy, True)

                            # Update all the turbines
                            # O(N) complexity
                            for k in range(self.n_turbines):
                                # Update the efficiency of the other turbines realitive to this turbine
                                self.turbines[k].update_distance(self.turbines[i], True)
                                # Update the efficiency of this turbine
                                self.turbines[i].update_distance(self.turbines[k], True)
                            
                            # Check if the current efficiency is better than the n_best
                            new_efficiency = self.efficiency()
                            for f in range(n_best):
                                if new_efficiency > new_nbest_efficiencies[f]:
                                    print("found improvement")
                                    new_nbest_efficiencies[f] = new_efficiency
                                    new_nbest_x[f] = pointx
                                    new_nbest_y[f] = pointy
                                    break
                            
                            # Revert the location of the turbines
                            for k in range(self.n_turbines):
                                self.turbines[k].revert_state()
                    # Update the last n_best
                    last_nbest_efficiencies = new_nbest_efficiencies
                    last_nbest_x = new_nbest_x
                    last_nbest_y = new_nbest_y
                    best_index = np.argmax(new_nbest_efficiencies)
                    self.turbines[i].update_location(new_nbest_x[best_index], new_nbest_y[best_index], False)
                    self.update_all_distances()

                    # Update the sigma
                    sigma2 = sigma2 * .5
                    #print(self.efficiency())
                # Reset sigma
                print(self.efficiency())
                sigma2 = sigma1
            # Update the overall sigma
            sigma1 = sigma1 * .5
        print(f"Final cross-entropy efficiency: {self.efficiency()}")
                
                
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
        self.min_distance_index = None
        self.last_min_distance = np.inf
        self.last_min_distance_index = None
        self.distances[i] = np.inf
        self.last_x = location[0]
        self.last_y = location[1]
        self.perturbed = False
    
    def update_location(self, x, y, perturbed):
        self.x = x
        self.y = y
        self.location = np.array([x, y])
        if perturbed:
            self.last_x = x
            self.last_y = y
            self.perturbed = True
    
    # Update the distance to a specific turbine
    def update_distance(self, turbine, perturbed):
        if not turbine.i == self.i:
            # Update the distance to the turbine
            new_distance = np.linalg.norm(self.location - turbine.location)
            self.distances[turbine.i] = new_distance
            # Case where the new distance is larger than the current minimum distance
            if((self.min_distance_index == turbine.i) and (new_distance > self.min_distance)):
                new_min_index = np.argmin(self.distances)
                self.last_min_distance = self.min_distance
                self.last_min_distance_index = self.min_distance_index
                self.min_distance = self.distances[new_min_index]
                self.min_distance_index = new_min_index
                self.perturbed = perturbed
            # Case where the new distance is smaller than the current minimum distance
            elif (new_distance < self.min_distance):
                self.last_min_distance = self.min_distance
                self.last_min_distance_index = self.min_distance_index
                self.min_distance = new_distance
                self.min_distance_index = turbine.i
                self.perturbed = perturbed

    def revert_state(self):
        if self.perturbed:
            self.min_distance = self.last_min_distance
            self.min_distance_index = self.last_min_distance_index
            self.x = self.last_x
            self.y = self.last_y
            self.perturbed = False
    
    # Returns the distance to the closest turbine and its index
    def return_min_distance(self):
        return self.min_distance, self.min_distance_index
    
    # Returns the efficiency of the turbine
    def efficiency(self):
        # Avoid division by zero
        return 1 / (1 + (1/(self.min_distance + epsilon)))
    
        
    
    

