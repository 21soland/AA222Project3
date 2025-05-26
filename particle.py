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
        self.vx = 0  # Separate velocity for x
        self.vy = 0  # Separate velocity for y
    
    def get_location(self):
        return self.x, self.y
    
    def update_location(self, overall_best_x, overall_best_y):
        # Update velocities for x and y separately
        self.vx = (self.w * self.vx + 
                  self.c1 * np.random.rand() * (self.best_x - self.x) + 
                  self.c2 * np.random.rand() * (overall_best_x - self.x))
        
        self.vy = (self.w * self.vy + 
                  self.c1 * np.random.rand() * (self.best_y - self.y) + 
                  self.c2 * np.random.rand() * (overall_best_y - self.y))
        
        # Update positions
        self.x += self.vx
        self.y += self.vy
    
    def get_efficiency(self):
        return self.efficiency
    
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
        self.x = x_rotated + center_x
        self.y = y_rotated + center_y

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

