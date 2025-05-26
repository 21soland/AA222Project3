import numpy as np


class Particle:
    def __init__(self, turbine_x, turbine_y, efficiency, w, c1, c2):
        self.x = turbine_x
        self.y = turbine_y
        self.efficiency = efficiency
        self.best_x = turbine_x
        self.best_y = turbine_y
        self.best_efficiency = efficiency
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.x_best = turbine_x
        self.y_best = turbine_y 
        self.v = 0
        self.best_efficiency = efficiency
    
    def get_location(self):
        return self.x, self.y
    
    def update_location(self, overall_best_x, overall_best_y):
        distance_to_local_best = np.linalg.norm(np.array([self.x, self.y]) - np.array([self.best_x, self.best_y]))
        distance_to_global_best = np.linalg.norm(np.array([self.x, self.y]) - np.array([overall_best_x, overall_best_y]))
        self.v = self.w * self.v + self.c1 * np.random.rand() * (distance_to_local_best) + self.c2 * np.random.rand() * (distance_to_global_best)
        self.x = self.x + self.v
        self.y = self.y + self.v
    
    def get_efficiency(self):
        return self.efficiency
    
    def set_efficiency(self, efficiency):
        self.efficiency = efficiency
        if efficiency > self.best_efficiency:
            self.best_efficiency = efficiency
            self.best_x = self.x
            self.best_y = self.y

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
        if (x - 5)**2 + (y - 5)**2 < 2**2:
            return False
        if (x)**2 + (y)**2 < 1**2:
            return False
        if (x)**2 + (y - 10)**2 < 1**2:
            return False
            
        return True

