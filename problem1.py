import numpy as np
from windfarm import WindFarm, Turbine

#np.random.seed(0)

def optimize(n_turbines, n_iterations):
    raise NotImplementedError("Not implemented")

# 
def optimize(n_turbines):
    # Start with brute forcing the best configuration
    windfarm = WindFarm(n_turbines)
    best_efficiency = windfarm.efficiency()
    print(f"Initial wind farm efficiency: {best_efficiency}")
    best_configuration = windfarm.turbines
    brute_force_iterations = 2
    # Brute force the best configuration
    for _ in range(brute_force_iterations):
        # Redistribute the turbines
        windfarm.redistribute()
        # Update the distances
        windfarm.update_all_distances()
        # Check if the efficiency is better
        if windfarm.efficiency() > best_efficiency:
            best_efficiency = windfarm.efficiency()
            best_configuration = windfarm.turbines
    windfarm.set_turbines(best_configuration)
    print(f"Brute force wind farm efficiency: {best_efficiency}")

    # Cross-entropy optimize the turbines
    n_iterations = 5 # How many times it runs cross-entropy on each point
    samples = 1000 # How many points it samples
    cycles = 10 # How many times it will cycle through all of the points
    windfarm.cross_entropy_optimize(n_iterations, samples, cycles)

    return windfarm

def main():
    n_turbines = 9
    windfarm = optimize(n_turbines)
    windfarm.plot()

if __name__ == "__main__":
    main()
