import numpy as np

EPSILON = 1e-6

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
        self.n_best_last_efficiencies = None
        self.n_best_last_stds = None
        self.n_best_last_x = None
        self.n_best_last_y = None
        
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
    
    def set_n_best_last(self, efficiencies, stds, x, y):
        """Set the last n best points."""
        self.n_best_last_efficiencies = efficiencies
        self.n_best_last_stds = stds
        self.n_best_last_x = x
        self.n_best_last_y = y
    
    def get_n_best_last(self):
        """Get the last n best points."""
        return self.n_best_last_efficiencies, self.n_best_last_stds, self.n_best_last_x, self.n_best_last_y
    
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
    
    def diff_from_mean(self,avg_dist, std):
        """
        Calculate the difference from the average distance.
        """
        return abs(self.temp_min_distance - avg_dist) / std

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
        
    
    

