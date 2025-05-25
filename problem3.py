from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib import pyplot as plt
import numpy as np

electricity_price = 0.33
power_per_turbine = 3300 * 8670 * 0.75
cost_per_turbine = 5400000

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

def calc_efficiency(n):
    distances = [13.4536, 10.3527, 9.114, 6.471, 5.8309, 5.2969, 5.01848, 4.44476, 4.2014]
    efficiencies = np.zeros(len(n))
    for i in range(len(n)):
        efficiencies[i] = 1/(1 + (1/distances[i]))
    return efficiencies


def profit(efficiency,n):
    prof = (n * efficiency * power_per_turbine * electricity_price) - (n * cost_per_turbine)
    return prof

def get_measured_efficiency():
    measured_efficiency = np.array([0.7157, 0.7178, 0.8497, 0.8768, 0.9218, 0.9344, 0.9538, 0.9472])
    min_seperation = np.array([3.0, 3.0, 6.0, 6.0, 12.0, 12.0, 15.0, 15.0])
    theoretical_efficiency = np.zeros(len(measured_efficiency))
    for i in range(len(measured_efficiency)):
        theoretical_efficiency[i] = 1/(1 + (1/min_seperation[i]))
    return theoretical_efficiency


def model_profit(gp,x):
    print(x)
    mean_residual, sigmas = gp.predict(x.reshape(-1, 1), return_std=True)
    final_efficiency = mean_residual
    print(f"Final efficiency: {final_efficiency}")
    modeled_profit = profit(final_efficiency,x)
    return modeled_profit

def get_confidence_interval(gp,x):
    mean_residual, sigmas = gp.predict(x.reshape(-1, 1), return_std=True)
    confidence_interval = (1.96 * sigmas)
    final_efficiency = mean_residual
    top_profit = profit(final_efficiency + confidence_interval,x)
    bottom_profit = profit(final_efficiency - confidence_interval,x)
    return top_profit, bottom_profit

def find_best_config(gp):
    max_profit = 0
    best_config = 0
    x = np.linspace(2, 10, 9)
    profit_vals = profit(calc_efficiency(x),x)
    best_config = x[profit_vals.argmax()]
    max_profit = profit_vals.max()
    print(f"Best configuration: {best_config} with profit: {max_profit}")

def theoretical_profit(n):
    theoretical_efficiencies = calc_efficiency(n)
    print(f"calc efficiency: {theoretical_efficiencies}")
    return profit(theoretical_efficiencies,n)

def plot_profit(gp):
    x1 = np.linspace(2, 10,9)
    x2 = np.linspace(2, 10, 100)
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
    find_best_config(gp)
    plot_profit(gp)

if __name__ == "__main__":
    main()


