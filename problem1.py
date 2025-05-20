import numpy as np
from windfarm import WindFarm

def main():
    n_turbines = 10
    locationsx= np.random.rand(n_turbines, 2) * 10
    locationsy = np.random.rand(n_turbines, 2) * 10
    locations = np.concatenate((locationsx, locationsy), axis=1)
    windfarm = WindFarm(n_turbines, locations)
    windfarm.plot()

if __name__ == "__main__":
    main()
