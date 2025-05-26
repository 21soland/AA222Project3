from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib import pyplot as plt
import numpy as np

electricity_price = 0.33
power_per_turbine = 3300 * 8670 * 0.75
cost_per_turbine = 5400000

# Get the the second column of the csv file
distances = np.genfromtxt('results.csv', delimiter=',', usecols=(1))

def dist_from_n(n):
    d = np.zeros(len(n))
    for i in range(len(n)):
        d[i] = distances[i]
    return d

def generate_gp():
        # Save the measured data
    min_seperation = np.array([3.0, 3.0, 6.0, 6.0, 12.0, 12.0, 15.0, 15.0])
    measured_efficiency = np.array([0.7157, 0.7178, 0.8497, 0.8768, 0.9218, 0.9344, 0.9538, 0.9472])
    og_measured_efficiency = measured_efficiency.copy()
    og_min_seperation = min_seperation.copy()

    std_deviation = 0
    for i in range(len(measured_efficiency)):
        theoretical_efficiency = 1/(1 + (1/min_seperation[i]))
        std_deviation += (measured_efficiency[i] - theoretical_efficiency)**2

    std_deviation = np.sqrt(std_deviation/len(measured_efficiency))

    # Fit the GP
    kernel = RBF(length_scale=8.0, length_scale_bounds="fixed") + WhiteKernel(noise_level=.01**2, noise_level_bounds="fixed")
    gp = GaussianProcessRegressor(kernel=kernel, alpha = 0, n_restarts_optimizer=8)
    gp.fit(min_seperation.reshape(-1, 1), measured_efficiency.reshape(-1, 1))
    return gp

def calc_efficiency(d):
    return 1 / (1 + (1/d))


def profit(efficiency,n):
    prof = (n * efficiency * power_per_turbine * electricity_price) - (n * cost_per_turbine)
    return prof


def model_profit(gp,x):
    d = dist_from_n(x)
    mean_residual, sigmas = gp.predict(d.reshape(-1, 1), return_std=True)
    final_efficiency = mean_residual
    modeled_profit = profit(final_efficiency,x)
    return modeled_profit

def get_confidence_interval(gp,x):
    d = dist_from_n(x)
    mean_residual, sigmas = gp.predict(d.reshape(-1, 1), return_std=True)
    confidence_interval = (1.96 * sigmas)
    final_efficiency = mean_residual
    top_profit = profit(final_efficiency + confidence_interval,x)
    bottom_profit = profit(final_efficiency - confidence_interval,x)
    return top_profit, bottom_profit

def find_best_config(x, modeled_profit):
    best_config = x[modeled_profit.argmax()]
    print(f"Best configuration: {best_config} with profit: {modeled_profit.max()}")

def theoretical_profit(x):
    d = dist_from_n(x)
    theoretical_efficiencies = calc_efficiency(d)
    return profit(theoretical_efficiencies,x)



def plot_profit(gp):
    x1 = np.linspace(2, 10,9)
    modeled_profit = model_profit(gp,x1)
    theo_profit = theoretical_profit(x1)

    upper_confidence, lower_confidence = get_confidence_interval(gp,x1)
    plt.fill_between(
        x1,
        upper_confidence,
        lower_confidence,
        color="tab:orange",
        alpha=0.5,
        label=r"95% Confidence Interval",
    )
    plt.plot(x1, modeled_profit, label='Modeled Profit')
    plt.plot(x1, theo_profit, label='Theoretical Profit',linestyle='--',color='red')
    min_profit = lower_confidence.min()
    max_profit = upper_confidence.max()
    margin = (max_profit - min_profit) * 0.1
    plt.xlabel('Number of turbines')
    plt.ylabel('Profit')
    plt.xlim(2, 10)
    plt.ylim(min_profit - margin, max_profit + margin)
    plt.title('Profit vs Number of Turbines')
    plt.legend()
    plt.show()

def main():
    gp = generate_gp()
    x = np.linspace(2, 10, 9)
    modeled_profit = model_profit(gp,x)
    find_best_config(x, modeled_profit)
    plot_profit(gp)

if __name__ == "__main__":
    main()


