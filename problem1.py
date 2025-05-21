import numpy as np
from windfarm import WindFarm, Turbine

def main():
    n_turbines = 10
    locations= np.random.rand(n_turbines, 2) * 10
    turbines = [Turbine(location, n_turbines, i) for i, location in enumerate(locations)]
    windfarm = WindFarm(turbines)
    print(windfarm.efficiency())
    windfarm.plot()

def evenly_distribute(n_turbines):
    raise NotImplementedError("Not implemented")

def optimize(n_turbines, n_iterations):
    raise NotImplementedError("Not implemented")

if __name__ == "__main__":
    main()
