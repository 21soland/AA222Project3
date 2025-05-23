import numpy as np
import matplotlib.pyplot as plt

# Small constant to avoid division by zero
EPSILON = 1e-6

class WindFarm:
    """
    Class representing a wind farm with multiple turbines.
    Handles turbine placement, optimization, and visualization.
    """
    def __init__(self, n_turbines):
        """
        Initialize a wind farm with n_turbines.
        
        Args:
            n_turbines (int): Number of turbines in the wind farm
        """
        self.turbines = None
        self.n_turbines = n_turbines

        # Turbine placement constraints
        self.x1, self.y1 = (5, 5)
        self.radius1 = 2
        self.x2, self.y2 = (0, 0)
        self.radius2 = 1
        self.x3, self.y3 = (0, 10)
        self.last_sigma1 = 10

        # Initialize turbine positions
        self.redistribute()
        self.update_all_distances()

    def efficiency(self):
        """
        Calculate total efficiency of all turbines.
        
        Returns:
            float: Total efficiency of the wind farm
        """
        return sum(turbine.efficiency() for turbine in self.turbines)
    
    def temp_efficiency(self):
        """
        Calculate total efficiency using temporary positions.
        
        Returns:
            float: Total temporary efficiency of the wind farm
        """
        return sum(turbine.temp_efficiency() for turbine in self.turbines)
    
    def redistribute(self):
        """
        Redistribute turbines using a randomized Latin square method.
        Ensures turbines are placed within constraints.
        """
        self.turbines = []
        permutation = np.random.permutation(self.n_turbines)
        grid_size = 10
        scale_factor = grid_size / self.n_turbines

        for n in range(self.n_turbines):
            # Use randomized Latin square method
            x = round((np.random.random() + n) * scale_factor, 2)
            y = round((np.random.random() + permutation[n]) * scale_factor, 2)

            # Ensure point lies within constraints
            while not self.check_constraints(x, y):
                x = np.random.random() * grid_size
                y = np.random.random() * grid_size

            self.turbines.append(Turbine(np.array([x, y]), self.n_turbines, n))

    def random_redistribute(self):
        """
        Redistribute turbines completely randomly within constraints.
        """
        self.turbines = []
        for i in range(self.n_turbines):
            x = np.random.random() * 10
            y = np.random.random() * 10
            while not self.check_constraints(x, y):
                x = np.random.random() * 10
                y = np.random.random() * 10
            self.turbines.append(Turbine(np.array([x, y]), self.n_turbines, i))
    
    def random_move(self, turbine):
        """
        Move a single turbine to a random valid position.
        
        Args:
            turbine (Turbine): The turbine to move
        """
        x = np.random.random() * 10
        y = np.random.random() * 10
        while not self.check_constraints(x, y):
            x = np.random.random() * 10
            y = np.random.random() * 10
        turbine.update_location(x, y)
    
    def average_min_distance(self):
        """
        Calculate average minimum distance between turbines.
        
        Returns:
            float: Average minimum distance
        """
        return sum(turbine.actual_min_distance for turbine in self.turbines) / self.n_turbines
    
    def check_constraints(self, x, y):
        """
        Check if a point (x,y) lies within all constraints.
        
        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            
        Returns:
            bool: True if point is valid, False otherwise
        """
        # Check boundary constraints
        if x <= 0 or x >= 10 or y <= 0 or y >= 10:
            return False
            
        # Check circular constraints
        if (x - self.x1)**2 + (y - self.y1)**2 < self.radius1**2:
            return False
        if (x - self.x2)**2 + (y - self.y2)**2 < self.radius2**2:
            return False
        if (x - self.x3)**2 + (y - self.y3)**2 < self.radius2**2:
            return False
            
        return True
    
    def update_all_distances(self):
        """
        Update distances between all pairs of turbines.
        O(N^2) complexity.
        """
        for i in range(self.n_turbines):
            for j in range(self.n_turbines):
                if i != j:
                    self.turbines[i].update_actual_distance(self.turbines[j])

    def set_turbines(self, turbines):
        """
        Set the turbine configuration and update distances.
        
        Args:
            turbines (list): List of Turbine objects
        """
        self.turbines = turbines
        self.update_all_distances()

    def expand_turbines_outwards(self, scale_factor):
        """
        Expand turbine positions outward from center.
        
        Args:
            scale_factor (float): Factor to scale distances by
        """
        center_x, center_y = 5, 5
        for turbine in self.turbines:
            new_x = (turbine.x - center_x) * scale_factor + center_x
            new_y = (turbine.y - center_y) * scale_factor + center_y
            turbine.update_location(new_x, new_y)
        self.update_all_distances()
    
    def rotate_turbines(self, angle):
        """
        Rotate all turbines around center point.
        
        Args:
            angle (float): Rotation angle in radians
        """
        center_x, center_y = 5, 5
        for turbine in self.turbines:
            new_x = (turbine.x - center_x) * np.cos(angle) - (turbine.y - center_y) * np.sin(angle) + center_x
            new_y = (turbine.x - center_x) * np.sin(angle) + (turbine.y - center_y) * np.cos(angle) + center_y
            turbine.update_location(new_x, new_y)
        self.update_all_distances()

    def reset_sigma1(self):
        """Reset the primary sigma value to initial state."""
        self.last_sigma1 = 10
    
    def reset_temp_distances(self):
        """Reset all temporary distances to actual distances."""
        for turbine in self.turbines:
            turbine.reset_temp_distances()
    
    def cross_entropy_optimize(self, n_iterations, samples, cycles, decay1, decay2, random_prob, verbose):
        """
        Optimize turbine positions using cross-entropy method.
        
        Args:
            n_iterations (int): Number of iterations per point
            samples (int): Number of sample points per iteration
            cycles (int): Number of optimization cycles
            decay1 (float): Cycle decay rate
            decay2 (float): Point decay rate
            random_prob (float): Probability of random movement
            verbose (bool): Whether to print progress
        """
        n_best = 3
        sigma1 = 3
        sigma2 = sigma1

        if verbose:
            print(f"Starting cross-entropy optimization with {n_iterations} iterations, "
                  f"{samples} samples, and {cycles} cycles")
            print(f"Initial cross-entropy efficiency: {self.efficiency()}")

        for cycle in range(cycles):
            if verbose:
                print(f"Cycle {cycle} of {cycles}, Current efficiency: {self.efficiency()}")

            for i in range(self.n_turbines):
                # Initialize best points tracking
                last_nbest_efficiencies = np.zeros(n_best)
                last_nbest_x = np.zeros(n_best)
                last_nbest_y = np.zeros(n_best)

                # Set initial values
                self.update_all_distances()
                self.reset_temp_distances()
                current_efficiency = self.efficiency()
                last_nbest_efficiencies.fill(current_efficiency)
                last_nbest_x.fill(self.turbines[i].x)
                last_nbest_y.fill(self.turbines[i].y)

                new_nbest_efficiencies = last_nbest_efficiencies.copy()
                new_nbest_x = last_nbest_x.copy()
                new_nbest_y = last_nbest_y.copy()

                # Randomly increase sigma for exploration
                if np.random.random() < random_prob:
                    sigma2 = max(sigma2 * 10, 0.2)

                for q in range(n_iterations):
                    # Generate sample points
                    x_samples = np.random.normal(0, sigma2, samples)
                    y_samples = np.random.normal(0, sigma2, samples)

                    for j in range(samples):
                        pointx = x_samples[j] + last_nbest_x[j % n_best]
                        pointy = y_samples[j] + last_nbest_y[j % n_best]

                        if self.check_constraints(pointx, pointy):
                            # Try new position
                            self.turbines[i].update_temp_location(pointx, pointy)

                            # Update distances
                            for k in range(self.n_turbines):
                                self.turbines[k].update_temp_distance(self.turbines[i])
                                self.turbines[i].update_temp_distance(self.turbines[k])
                            
                            # Check if improvement found
                            new_efficiency = self.temp_efficiency()
                            for f in range(n_best):
                                if new_efficiency > new_nbest_efficiencies[f]:
                                    new_nbest_efficiencies[f] = new_efficiency
                                    new_nbest_x[f] = pointx
                                    new_nbest_y[f] = pointy
                                    break
                            
                            self.reset_temp_distances()

                    # Update best points
                    last_nbest_efficiencies = new_nbest_efficiencies
                    last_nbest_x = new_nbest_x
                    last_nbest_y = new_nbest_y
                    best_index = np.argmax(new_nbest_efficiencies)
                    self.turbines[i].update_location(new_nbest_x[best_index], new_nbest_y[best_index])
                    self.update_all_distances()

                    # Update sigma
                    sigma2 = sigma2 * decay2

                # Reset sigma for next turbine
                sigma2 = sigma1

            # Update overall sigma
            if np.random.random() < 0.7:
                sigma1 = sigma1 * decay1

            if verbose:
                print(f"Sigma1: {sigma1}")
                self.print_cur_error()

            self.reset_temp_distances()

        if verbose:
            print(f"Final cross-entropy efficiency: {self.efficiency()}")
            self.print_min_distances()
    
    def print_min_distances(self):
        """Print minimum distances for all turbines."""
        for turbine in self.turbines:
            print(f"Turbine {turbine.i} has a minimum distance of {turbine.actual_min_distance}")
                
    def print_cur_error(self):
        """Print current error compared to known solutions."""
        answers = [13.4536, 10.3527, 9.114, 6.471, 5.8309, 5.2969, 5.01848, 4.44476, 4.2014]
        error = abs(self.average_min_distance() - answers[self.n_turbines-2])
        print(f"Current error: {error}")
    
    def cur_error(self):
        """
        Calculate current error compared to known solutions.
        
        Returns:
            float: Absolute error
        """
        answers = [13.4536, 10.3527, 9.114, 6.471, 5.8309, 5.2969, 5.01848, 4.44476, 4.2014]
        return abs(self.average_min_distance() - answers[self.n_turbines-2])
    
    def dist_std(self):
        """
        Calculate standard deviation of minimum distances.
        
        Returns:
            float: Standard deviation of minimum distances
        """
        average = self.average_min_distance()
        std = sum((turbine.actual_min_distance - average)**2 for turbine in self.turbines)
        return np.sqrt(std / self.n_turbines)

    def plot(self):
        """Plot the wind farm layout with turbines and constraints."""
        # Plot turbines
        for turbine in self.turbines:
            plt.scatter(turbine.x, turbine.y)
            
        # Plot constraints
        circle1 = plt.Circle((self.x1, self.y1), self.radius1, fill=False)
        circle2 = plt.Circle((self.x2, self.y2), self.radius2, fill=False)
        circle3 = plt.Circle((self.x3, self.y3), self.radius2, fill=False)
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.gca().add_patch(circle3)
        
        # Set plot properties
        plt.axis('equal')
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.show()
    
    def __del__(self):
        """Clean up plot resources."""
        plt.close()


class Turbine:
    """
    Class representing a single wind turbine.
    Handles position and distance calculations.
    """
    def __init__(self, location, n_turbines, i):
        """
        Initialize a turbine.
        
        Args:
            location (np.array): Initial position [x, y]
            n_turbines (int): Total number of turbines
            i (int): Index of this turbine
        """
        self.location = location
        self.x = location[0]
        self.y = location[1]
        self.temp_location = location
        self.i = i
        
        # Initialize distance arrays
        self.actual_distances = np.zeros(n_turbines)
        self.temp_distances = np.zeros(n_turbines)
        self.actual_distances[i] = np.inf
        self.temp_distances[i] = np.inf
        
        # Initialize minimum distance tracking
        self.actual_min_distance = np.inf
        self.actual_min_distance_index = None
        self.temp_min_distance = np.inf
        self.temp_min_distance_index = None
    
    def update_location(self, x, y):
        """
        Update both actual and temporary locations.
        
        Args:
            x (float): New x-coordinate
            y (float): New y-coordinate
        """
        self.x = x
        self.y = y
        self.location = np.array([x, y])
        self.temp_location = np.array([x, y])
    
    def update_temp_location(self, x, y):
        """
        Update only temporary location.
        
        Args:
            x (float): New x-coordinate
            y (float): New y-coordinate
        """
        self.temp_location = np.array([x, y])

    def update_actual_distance(self, turbine):
        """
        Update actual distance to another turbine.
        
        Args:
            turbine (Turbine): Other turbine to measure distance to
        """
        if not turbine.i == self.i:
            new_distance = np.linalg.norm(self.location - turbine.location)
            self.actual_distances[turbine.i] = new_distance
            min_index = np.argmin(self.actual_distances)
            self.actual_min_distance = self.actual_distances[min_index]
            self.actual_min_distance_index = min_index
    
    def reset_temp_distances(self):
        """Reset temporary distances to match actual distances."""
        self.temp_distances = self.actual_distances.copy()
        self.temp_min_distance = self.actual_min_distance
        self.temp_min_distance_index = self.actual_min_distance_index
        self.temp_location = self.location

    def update_temp_distance(self, turbine):
        """
        Update temporary distance to another turbine.
        
        Args:
            turbine (Turbine): Other turbine to measure distance to
        """
        if not turbine.i == self.i:
            new_distance = np.linalg.norm(self.temp_location - turbine.location)
            self.temp_distances[turbine.i] = new_distance
            
            # Update minimum distance if needed
            if (self.temp_min_distance_index == turbine.i) and (new_distance > self.temp_min_distance):
                new_min_index = np.argmin(self.temp_distances)
                self.temp_min_distance = self.temp_distances[new_min_index]
                self.temp_min_distance_index = new_min_index
            elif new_distance < self.temp_min_distance:
                self.temp_min_distance = new_distance
                self.temp_min_distance_index = turbine.i
    
    def efficiency(self):
        """
        Calculate turbine efficiency based on minimum distance.
        
        Returns:
            float: Turbine efficiency
        """
        return 1 / (1 + (1/(self.actual_min_distance + EPSILON)))
    
    def temp_efficiency(self):
        """
        Calculate temporary turbine efficiency.
        
        Returns:
            float: Temporary turbine efficiency
        """
        return 1 / (1 + (1/(self.temp_min_distance + EPSILON)))
        
    
    

