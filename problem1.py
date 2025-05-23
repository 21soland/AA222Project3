import numpy as np
from windfarm import WindFarm, Turbine
import time

# Known optimal solutions for different numbers of turbines
ANSWERS = [13.4536, 10.3527, 9.114, 6.471, 5.8309, 5.2969, 5.01848, 4.44476, 4.2014]

def optimize(n_turbines, n_iterations):
    """
    Legacy optimization function (not implemented)
    """
    raise NotImplementedError("Not implemented")


def optimize(n_turbines):
    """
    Main optimization function for wind farm layout.
    
    Args:
        n_turbines (int): Number of turbines to optimize
        
    Returns:
        tuple: (efficiency, average_min_distance)
    """
    # Initialize wind farm and track best configuration
    windfarm = WindFarm(n_turbines)
    best_efficiency = windfarm.efficiency()
    best_dist_std = windfarm.dist_std()
    print(f"Initial wind farm efficiency: {best_efficiency}")
    best_configuration = windfarm.turbines
    
    # Brute force optimization parameters
    brute_force_iterations = 500
    
    # Brute force optimization loop
    for iteration in range(brute_force_iterations):
        # Reset and redistribute turbines
        windfarm.reset_sigma1()
        windfarm.redistribute()
        
        # Quick optimization pass
        windfarm.cross_entropy_optimize(
            n_iterations=2,
            samples=3,
            cycles=3,
            decay1=0.5,
            decay2=0.25,
            random_prob=0.5,
            verbose=False
        )
        
        # Update best configuration if current is better
        cur_std = windfarm.dist_std()
        if cur_std < best_dist_std:
            best_dist_std = cur_std
            best_configuration = windfarm.turbines
            
        # Progress reporting
        if iteration % 100 == 0:
            print(f"Brute force iteration {iteration} of {brute_force_iterations}, "
                  f"best dist std: {best_dist_std}")
    
    # Set the best configuration found
    windfarm.set_turbines(best_configuration)
    print(f"Brute force wind farm efficiency: {best_efficiency}")

    # Final detailed optimization
    windfarm.cross_entropy_optimize(
        n_iterations=5,      # Number of iterations per point
        samples=2000,        # Number of sample points
        cycles=25,           # Number of optimization cycles
        decay1=0.9,          # Cycle decay rate
        decay2=0.8,          # Point decay rate
        random_prob=0.25,    # Probability of random movement
        verbose=True
    )
    windfarm.update_all_distances()
    
    return windfarm.efficiency(), windfarm.average_min_distance()


def main():
    """
    Main function to run optimization for different numbers of turbines
    and save results to CSV.
    """
    start_time = time.time()
    
    # Run optimization for 2-10 turbines
    for n_turbines in range(2, 11):
        # Run optimization
        efficiency, average_min_distance = optimize(n_turbines)
        
        # Save results to CSV
        with open('results.csv', 'a') as f:
            f.write(f"{n_turbines},{average_min_distance}\n")
            
        # Print results
        print(f"Wind farm efficiency for {n_turbines} turbines: {efficiency}")
        print(f"Average minimum distance for {n_turbines} turbines: {average_min_distance}")
        
        # Calculate and print error
        error = abs(efficiency - ANSWERS[n_turbines-2])
        print(f"Error for {n_turbines} turbines: {error}")
    
    # Print total execution time
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


def temp():
    """
    Temporary function for testing optimization with 8 turbines.
    """
    n_turbines = 8
    efficiency, average_min_distance = optimize(n_turbines)
    
    print(f"Average minimum distance: {average_min_distance}")
    error = abs(average_min_distance - ANSWERS[n_turbines-2])
    print(f"Error for {n_turbines} turbines: {error}")


if __name__ == "__main__":
    # main()
    temp()
