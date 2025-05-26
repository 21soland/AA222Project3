import numpy as np
import matplotlib.pyplot as plt
from turbine import Turbine
from particle import Particle

# Small constant to avoid division by zero
EPSILON = 0

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
        p_star = self.average_min_distance()
        return self.n_turbines / (1 + (1/(p_star + EPSILON)))
    
    def temp_efficiency(self):
        """
        Calculate total efficiency using temporary positions.
        
        Returns:
            float: Total temporary efficiency of the wind farm
        """
        p_star = self.average_min_distance_temp()
        return self.n_turbines / (1 + (1/(p_star + EPSILON)))
    
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
    
    def average_min_distance_temp(self):
        """
        Calculate average minimum distance between turbines.
        
        Returns:
            float: Average minimum distance
        """
        return sum(turbine.temp_min_distance for turbine in self.turbines) / self.n_turbines
    

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


    def reset_sigma1(self):
        """Reset the primary sigma value to initial state."""
        self.last_sigma1 = 10
    
    def reset_temp_distances(self):
        """Reset all temporary distances to actual distances."""
        for turbine in self.turbines:
            turbine.reset_temp_distances()
    
    def dist_std_temp(self,turbine):
        """Calculate standard deviation of temporary distances."""
        average = self.average_min_distance_temp()
        std = abs(turbine.temp_min_distance - average)
        return std
    
    def quick_update_turbines(self, turbine_index, new_x, new_y):
        # Try the temporary new position
        self.turbines[turbine_index].update_location(new_x, new_y)

        # Update the temporary distances
        for k in range(self.n_turbines):
            self.turbines[k].update_temp_distance(self.turbines[turbine_index])
            self.turbines[turbine_index].update_temp_distance(self.turbines[k])
    
    def worst_turbine_deviation(self):
        """
        Calculate the deviation of the worst turbine from the mean.
        """
        max_deviation = 0
        for turbine in self.turbines:
            deviation = turbine.diff_from_mean(self.average_min_distance())
            if deviation > max_deviation:
                max_deviation = deviation
        return max_deviation
    
    def early_stop(self):
        deviation = 0
        for turbine in self.turbines:
            deviation += turbine.diff_from_mean(self.average_min_distance())
        avg_deviation = deviation / self.n_turbines
        if avg_deviation < 1e-6:
            return True
        return False

    
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
        n_best = 5
        sigma1 = 3.0  # Increased initial sigma for better exploration
        sigma2 = sigma1

        if verbose:
            print(f"Starting cross-entropy optimization with {n_iterations} iterations, "
                  f"{samples} samples, and {cycles} cycles")
            print(f"Initial cross-entropy efficiency: {self.efficiency()}")

        for cycle in range(cycles):
            if verbose:
                print(f"\nCycle {cycle} of {cycles}: Current efficiency: {self.efficiency()}")

            # Update the distances once per cycle
            self.update_all_distances()
            
            # For each turbine  
            for i in range(self.n_turbines):
                # Reset sigma for next turbine
                sigma2 = sigma1

                # Scale sigma based on turbine's deviation from mean
                factor = 1.0
                if cycle > 0:  # Start scaling from first cycle
                    deviation = self.turbines[i].diff_from_mean(self.average_min_distance())
                    sigma2 = sigma2 * (1 + deviation * factor)
                    # Cap sigma2 to prevent overflow but allow reasonable exploration
                    sigma2 = min(max(sigma2, 1e-3), 2.0)

                # Initialize best points tracking
                last_nbest_efficiencies = np.zeros(n_best)
                last_nbest_stds = np.zeros(n_best)
                last_nbest_x = np.zeros(n_best)
                last_nbest_y = np.zeros(n_best)

                # Set initial values
                self.reset_temp_distances()
                current_efficiency = self.efficiency()
                current_std = self.dist_std(self.turbines[i])
                last_nbest_efficiencies.fill(current_efficiency)
                last_nbest_stds.fill(current_std)
                last_nbest_x.fill(self.turbines[i].x)
                last_nbest_y.fill(self.turbines[i].y)

                new_nbest_efficiencies = last_nbest_efficiencies.copy()
                new_nbest_stds = last_nbest_stds.copy()
                new_nbest_x = last_nbest_x.copy()
                new_nbest_y = last_nbest_y.copy()

                for q in range(n_iterations):
                    # Generate sample points with increased diversity
                    x_samples = np.random.normal(0, sigma2, samples)
                    y_samples = np.random.normal(0, sigma2, samples)
                    
                    """# Add some random samples for better exploration
                    if np.random.random() < random_prob:
                        x_samples[:samples//4] = np.random.uniform(-sigma2*2, sigma2*2, samples//4)
                        y_samples[:samples//4] = np.random.uniform(-sigma2*2, sigma2*2, samples//4)"""

                    for j in range(samples):
                        # Generate a random point within the constraints
                        pointx = x_samples[j] + last_nbest_x[j % n_best]
                        pointy = y_samples[j] + last_nbest_y[j % n_best]
                        
                        # If point is invalid, try with increasing sigma
                        max_attempts = 3
                        attempt = 0
                        while not self.check_constraints(pointx, pointy) and attempt < max_attempts:
                            # Increase sigma for each attempt
                            attempt_sigma = sigma2 * (1.5 ** attempt)  # More gradual increase
                            pointx = np.random.normal(last_nbest_x[j % n_best], attempt_sigma)
                            pointy = np.random.normal(last_nbest_y[j % n_best], attempt_sigma)
                            attempt += 1
                        
                        # If still invalid, use a random point but keep it close to the best point
                        if not self.check_constraints(pointx, pointy):
                            pointx = last_nbest_x[j % n_best] + np.random.normal(0, sigma2)
                            pointy = last_nbest_y[j % n_best] + np.random.normal(0, sigma2)
                            while not self.check_constraints(pointx, pointy):
                                pointx = last_nbest_x[j % n_best] + np.random.normal(0, sigma2)
                                pointy = last_nbest_y[j % n_best] + np.random.normal(0, sigma2)
                        
                        # Try the temporary new position
                        self.turbines[i].update_temp_location(pointx, pointy)

                        # Update the temporary distances
                        for k in range(self.n_turbines):
                            self.turbines[k].update_temp_distance(self.turbines[i])
                            self.turbines[i].update_temp_distance(self.turbines[k])
                        
                        # Check if the temporary new position is better than the best points
                        new_efficiency = self.temp_efficiency()
                        new_std = self.dist_std_temp(self.turbines[i])

                        # If so add it to the n best points
                        for k in range(n_best):
                            if new_efficiency > new_nbest_efficiencies[k]:
                                # Shift existing points
                                new_nbest_efficiencies[k+1:] = new_nbest_efficiencies[k:-1]
                                new_nbest_stds[k+1:] = new_nbest_stds[k:-1]
                                new_nbest_x[k+1:] = new_nbest_x[k:-1]
                                new_nbest_y[k+1:] = new_nbest_y[k:-1]
                                
                                # Insert new point
                                new_nbest_efficiencies[k] = new_efficiency
                                new_nbest_stds[k] = new_std
                                new_nbest_x[k] = pointx
                                new_nbest_y[k] = pointy
                                break
                        
                        # Reset the temporary distances
                        self.reset_temp_distances()

                    # Update best points
                    last_nbest_efficiencies = new_nbest_efficiencies
                    last_nbest_stds = new_nbest_stds
                    last_nbest_x = new_nbest_x
                    last_nbest_y = new_nbest_y
                    
                    self.turbines[i].update_location(last_nbest_x[0], last_nbest_y[0])
                    self.update_all_distances()

                    # Update sigma
                    sigma2 = sigma2 * decay2

            # Update the overall cycle sigma
            sigma1 = sigma1 * decay1

            if verbose:
                print(f"Sigma1: {sigma1}")
                self.print_avg_deviation()
                self.print_cur_error()
                
            if self.early_stop():
                if verbose:
                    print("Early stopping")
                break

            self.reset_temp_distances()

        if verbose:
            print(f"Final cross-entropy efficiency: {self.efficiency()}")
            self.print_min_distances()
    
    def particle_swarm_optimize(self, n_iterations, n_particles, cycles, w, c1, c2, verbose):
        """
        Optimize turbine positions using particle swarm optimization.
        """
        for cycle in range(cycles):
            self.update_all_distances()
            for i in range(self.n_turbines):
                # Get the initial turbine position and efficiency
                start_x = self.turbines[i].x
                start_y = self.turbines[i].y
                start_efficiency = self.efficiency()

                # Initialize the global best for the particle swarm
                cur_best_x = start_x
                cur_best_y = start_y
                cur_best_efficiency = start_efficiency

                # Initialize the particles with random positions around the current turbine
                particles = []
                for j in range(n_particles):
                    # Add random variation to initial position
                    init_x = start_x + np.random.normal(0, 0.5)
                    init_y = start_y + np.random.normal(0, 0.5)
                    
                    # Ensure initial position is within constraints
                    while not self.check_constraints(init_x, init_y):
                        init_x = start_x + np.random.normal(0, 0.5)
                        init_y = start_y + np.random.normal(0, 0.5)
                    
                    particles.append(Particle(init_x, init_y, start_efficiency, w, c1, c2))
                
                # Run particle swarm optimization
                prev_best_efficiency = cur_best_efficiency
                no_improvement_count = 0
                
                for n in range(n_iterations):
                    cur_best_x, cur_best_y, cur_best_efficiency = self.update_particles(particles, i, cur_best_x, 
                                                                                        cur_best_y, cur_best_efficiency)
                    
                    # Check for early stopping
                    if abs(cur_best_efficiency - prev_best_efficiency) < 1e-6:
                        no_improvement_count += 1
                        if no_improvement_count >= 5:  # Stop if no improvement for 5 iterations
                            break
                    else:
                        no_improvement_count = 0
                    
                    prev_best_efficiency = cur_best_efficiency
                
                # Set the turbine to the new best location
                self.turbines[i].update_location(cur_best_x, cur_best_y)

                # Update the overall efficiency
                self.update_all_distances()
                
            if verbose:
                print(f"Cycle {cycle} of {cycles}: Current efficiency: {self.efficiency()}")
                self.print_cur_error()
                print("\n")
                
        if verbose:
            print(f"Final efficiency: {self.efficiency()}")
            self.print_min_distances()
        
    def update_particles(self, particles, turbine_index, cur_best_x, cur_best_y, cur_best_efficiency):
        """
        Update the location of the particles based on the current best position.
        """
        new_best_efficiency = cur_best_efficiency
        new_best_x = cur_best_x
        new_best_y = cur_best_y
        
        for particle in particles:
            # Update the location of the particle using particle swarm algorithm
            particle.update_location(cur_best_x, cur_best_y)

            # Get the new location of the particle
            particle_x, particle_y = particle.get_location()
            
            # If the new position is invalid, try to find a valid position
            max_attempts = 3
            attempt = 0
            while not self.check_constraints(particle_x, particle_y) and attempt < max_attempts:
                # Reduce velocity and try again
                particle.vx *= 0.5
                particle.vy *= 0.5
                particle.x = particle.best_x + particle.vx
                particle.y = particle.best_y + particle.vy
                particle_x, particle_y = particle.get_location()
                attempt += 1
            
            # If still invalid, reset to best known position
            if not self.check_constraints(particle_x, particle_y):
                particle.x = particle.best_x
                particle.y = particle.best_y
                particle.vx = 0
                particle.vy = 0
                particle_x, particle_y = particle.get_location()

            # Update the temporary location of the turbine to the new location of the particle
            self.turbines[turbine_index].update_temp_location(particle_x, particle_y)

            # Update the temporary distances
            for k in range(self.n_turbines):
                self.turbines[k].update_temp_distance(self.turbines[turbine_index])
                self.turbines[turbine_index].update_temp_distance(self.turbines[k])
            
            # Get the efficiency of the new position
            new_efficiency = self.temp_efficiency()
            particle.set_efficiency(new_efficiency)

            # Reset the temporary distances
            self.reset_temp_distances()

            # If the new efficiency is better than the current best, update the current best
            if new_efficiency > new_best_efficiency:
                new_best_efficiency = new_efficiency
                new_best_x = particle_x
                new_best_y = particle_y
                
        # Return the new best location and efficiency
        return new_best_x, new_best_y, new_best_efficiency
                
                
    def print_min_distances(self):
        """Print minimum distances for all turbines."""
        for turbine in self.turbines:
            print(f"Turbine {turbine.i} has a minimum distance of {turbine.actual_min_distance}")
                
    def print_cur_error(self):
        """Print current error compared to known solutions."""
        answers = [13.4536, 10.3527, 9.114, 6.471, 5.8309, 5.2969, 5.01848, 4.44476, 4.2014]
        error = abs(self.average_min_distance() - answers[self.n_turbines-2])
        print(f"Current error: {error}")
    
    def print_avg_deviation(self):
        avg_deviation = 0
        for turbine in self.turbines:
            avg_deviation += turbine.diff_from_mean(self.average_min_distance())
        avg_deviation /= self.n_turbines
        print(f"Average deviation: {avg_deviation}")

    
    def cur_error(self):
        """
        Calculate current error compared to known solutions.
        
        Returns:
            float: Absolute error
        """
        answers = [13.4536, 10.3527, 9.114, 6.471, 5.8309, 5.2969, 5.01848, 4.44476, 4.2014]
        return abs(self.average_min_distance() - answers[self.n_turbines-2])
    
    def dist_std(self,turbine):
        """
        Calculate standard deviation of minimum distances.
        
        Returns:
            float: Standard deviation of minimum distances
        """
        average = self.average_min_distance()
        std = abs(turbine.actual_min_distance - average)
        return std

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
