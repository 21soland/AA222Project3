import numpy as np

EPSILON = 0

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
    
    def get_constraint(self,x,y):
        # Check boundary constraints
        if x <= 0 or x >= 10 or y <= 0 or y >= 10:
            return False
            
        # Check circular constraints
        if (x - 5)**2 + (y - 5)**2 < 4:
            return False
        if (x)**2 + (y - 0)**2 < 1:
            return False
        if (x)**2 + (y - 10)**2 < 1:
            return False
            
        return True

    def rotate(self, angle, center_x, center_y):
        """
        Rotate the particle around a center point.
        
        Args:
            angle (float): Angle to rotate in degrees
            center_x (float): x-coordinate of the center point
            center_y (float): y-coordinate of the center point
        """
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Translate coordinates to origin
        x_centered = self.x - center_x
        y_centered = self.y - center_y
        
        # Rotate coordinates
        x_rotated = x_centered * np.cos(angle_rad) - y_centered * np.sin(angle_rad)
        y_rotated = x_centered * np.sin(angle_rad) + y_centered * np.cos(angle_rad)
        
        # Translate back to original position
        new_x = x_rotated + center_x
        new_y = y_rotated + center_y
        
        # Find the closest point in bounds
        while not self.get_constraint(new_x, new_y):
            sigma = 0.5
            new_x = new_x + np.random.normal(0, sigma)
            new_y = new_y + np.random.normal(0, sigma)
            sigma *= 1.1
        self.x = new_x
        self.y = new_y
    
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
    
    def diff_from_mean(self,avg_dist):
        """
        Calculate the difference from the average distance.
        """
        return abs(self.temp_min_distance - avg_dist)

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
        
    
    

