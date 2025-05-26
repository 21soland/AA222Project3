import numpy as np
from windfarm import WindFarm, Turbine
import time
from matplotlib import pyplot as plt

def brute_force_optimize(n_turbines,brute_force_iterations,windfarm):
    best_efficiency = windfarm.efficiency()
    best_configuration = windfarm.turbines
    # Brute force optimization loop
    for iteration in range(brute_force_iterations):
        # Reset and redistribute turbines
        windfarm.reset_sigma1()
        windfarm.redistribute()
        
        # Quick optimization pass
        windfarm.cross_entropy_optimize(
            n_iterations=3,
            samples=10,
            cycles=3,
            decay1=0.75,
            decay2=0.5,
            random_prob=0.5,
            verbose=False
        )
        
        # Update best efficiency if current is better
        cur_efficiency = windfarm.efficiency()
        if cur_efficiency > best_efficiency:
            best_efficiency = cur_efficiency
            best_configuration = windfarm.turbines
        
            
        # Progress reporting
        if iteration % 100 == 0:
            print(f"Brute Force Iteration {iteration} of {brute_force_iterations}, "
                  f"Best Efficiency: {best_efficiency}")
        
        # Reset sigma
        windfarm.reset_sigma1()
    
    # Set the best configuration found
    windfarm.set_turbines(best_configuration)
    print(f"\nBrute force wind farm efficiency: {best_efficiency}")
    print(f"Average minimum distance: {windfarm.average_min_distance()}")

def cross_entropy_optimize(n_turbines):
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
    print(f"\nInitial wind farm efficiency: {best_efficiency}\n")
    
    # Brute force optimization parameters
    brute_force_iterations = 200
    brute_force_optimize(n_turbines,brute_force_iterations,windfarm)
    
    # Final detailed optimization
    windfarm.reset_sigma1()
    windfarm.cross_entropy_optimize(
        n_iterations=8,      # Number of iterations per point
        samples=2000,        # Number of sample points
        cycles=10,           # Number of optimization cycles
        decay1=0.5,          # Cycle decay rate
        decay2=0.5,          # Point decay rate
        random_prob=0.25,    # Probability of random rotation/expansion
        verbose=True
    )
    windfarm.update_all_distances()
    
    return windfarm.efficiency(), windfarm.average_min_distance()

def particle_swarm_optimize(n_turbines,cycles,graph):
    """
    Main optimization function for wind farm layout using particle swarm optimization.
    """
    windfarm = WindFarm(n_turbines)

    brute_force_iterations = 600

    # Run brute force optimization with cross entropy
    # This finds a good starting point for the particle swarm optimization
    brute_force_optimize(n_turbines,brute_force_iterations,windfarm)
    # Find the definitive global optimum with particle swarm optimization
    windfarm.particle_swarm_optimize(
        n_iterations=100, # How many times the particles will update their position
        n_particles=2000, # How many particles to use
        cycles=cycles, # How many times we cycle through each turbine
        w=.5, # Weight of the particle's current position
        c1=1, # Weight of the particle's best position
        c2=1, # Weight of the global best position
        rotate_prob=0.0, # Probability of rotating the turbines
        verbose=True)
    windfarm.update_all_distances()
    
    if graph:
        windfarm.plot()

    return windfarm.efficiency(), windfarm.average_min_distance()

def main():
    """
    Main function to run optimization for different numbers of turbines
    and save results to CSV.
    """
    # Read the previous results from the CSV file
    previous_results = []
    with open('results.csv', 'r') as f:
        for line in f:
            previous_results.append(line.strip().split(','))
    
    # Clear the results.csv file
    with open('results.csv', 'w') as f:
        f.write("")
    
    start_time = time.time()
    
    # Run optimization for 2-10 turbines
    for n_turbines in range(2, 11):
        # Run optimization
        cycles = 15
        # Add more cycles for configurations that need them
        if n_turbines == 6:
            cycles = 20
            
        efficiency, average_min_distance = particle_swarm_optimize(n_turbines,cycles,graph=False)
            
        # Print results
        print("\n**************************************************")
        print(f"Wind farm efficiency for {n_turbines} turbines: {efficiency}")
        print(f"Average minimum distance for {n_turbines} turbines: {average_min_distance}")
        print("**************************************************\n")

        # Update the CSV file if the efficiency is better than the previous best
        try:
            if average_min_distance > previous_results[n_turbines-2][1]:
                with open('results.csv', 'a') as f:
                    f.write(f"{n_turbines},{average_min_distance}\n")
            else:
                with open('results.csv', 'a') as f:
                    f.write(f"{n_turbines},{previous_results[n_turbines-2][1]}\n")
        except: # If the file is empty, write the current result
            with open('results.csv', 'a') as f:
                f.write(f"{n_turbines},{average_min_distance}\n")
    # Print total execution time
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

def temp_optimization():
    """
    Main function to run optimization for different numbers of turbines
    and save results to CSV.
    """
    
    start_time = time.time()
    
    n_turbines = 7
    # Run optimization
    cycles = 20
    # Add more cycles for configurations that need them
    efficiency, average_min_distance = particle_swarm_optimize(n_turbines,cycles,graph=True)
    
        
    # Print results
    print("\n**************************************************")
    print(f"Wind farm efficiency for {n_turbines} turbines: {efficiency}")
    print(f"Average minimum distance for {n_turbines} turbines: {average_min_distance}")
    print("**************************************************\n")
    
    # Print total execution time
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

def plot_min_distances():
    """
    Plot the minimum distances for the 2-10 turbines
    """
    n_turbines = np.arange(2, 11)
    distances = np.genfromtxt('results.csv', delimiter=',', usecols=(1))
    plt.figure(figsize=(10, 6))
    plt.plot(distances, n_turbines)
    plt.title(f'Minimum Distances for n Turbines')
    plt.xlabel('Minimum Distance')
    plt.ylabel('Number of Turbines')
    plt.show()

def plot_seperation_vs_efficiency():
    """
    Plot the seperation vs efficiency for the 2-10 turbines
    """
    x = np.linspace(0, 16, 100)
    efficiencies = 1 / (1 + (1/x))
    plt.figure(figsize=(10, 6))
    plt.plot(x, efficiencies)
    plt.title(f'Seperation vs Efficiency')
    plt.xlabel('Seperation Distance')
    plt.ylabel('Efficiency')
    plt.show()
    
    
if __name__ == "__main__":
    main()
    #plot_min_distances()
    #plot_seperation_vs_efficiency()
    #temp_optimization()
