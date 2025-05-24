from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


# Generate the ideal data
X = np.linspace(2, 15, 100).reshape(-1, 1)
Y = 1/(1 + (1 / X))

# Save the measured data
min_seperation = np.array([3.0, 3.0, 6.0, 6.0, 12.0, 12.0, 15.0, 15.0])
measured_efficiency = np.array([0.7157, 0.7178, 0.8497, 0.8768, 0.9218, 0.9344, 0.9538, 0.9472])
og_measured_efficiency = measured_efficiency.copy()
og_min_seperation = min_seperation.copy()

noise_std = 0.01
# Fit the GP
kernel = RBF(length_scale=8.0, length_scale_bounds=(1e-2, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=8)
gp.fit(min_seperation.reshape(-1, 1), measured_efficiency.reshape(-1, 1))

x = np.linspace(0, 15, 100)
means, sigmas = gp.predict(x.reshape(-1, 1), return_std=True)
confidence_interval = (1.96 * sigmas)
upper = means + confidence_interval
lower = means - confidence_interval

plt.plot(X, Y, color='red', linestyle='dashed', label='Ideal data')

plt.scatter(og_min_seperation, og_measured_efficiency, label='GP mean')
plt.plot(x, means, label='GP mean')
plt.fill_between(
    x,
    upper,
    lower,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)

plt.xlabel('Minimum seperation')
plt.ylabel('Efficiency')
plt.title('Gaussian Process Regression')
plt.xlim(3, 16)
plt.ylim(0.7, 1)
plt.legend()
plt.grid(True)
plt.show()