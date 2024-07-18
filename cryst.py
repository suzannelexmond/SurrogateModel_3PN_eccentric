import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example data
X_train = np.array([[0.0229], [0.1277], [0.1989], [0.1882], [0.1802], [0.0991], [0.1669], [0.1935], [0.1749], [0.1433], [0.1574], [0.1357], [0.0568], [0.0108], [0.1844], [0.1627], [0.1711], [0.1962], [0.1212], [0.1482], [0.1532], [0.0915], [0.1909], [0.1395]])
y_train = np.array([-1.48238859e-23, 2.46516797e-23, -1.36982893e-22, -1.00516515e-22, 8.52718551e-23, -3.79522217e-23, -5.10377962e-23, -1.48382765e-22, 1.15000815e-22, 1.73858850e-23, -1.16143057e-22, 9.51799095e-23, -7.17277420e-25, -7.16158644e-24, -2.18627604e-23, -9.99058800e-23, 3.37102917e-23, -1.50466586e-22, -4.15364637e-23, -5.49936368e-23, -1.01440847e-22, 1.02630152e-23, -1.32173728e-22, 7.28159867e-23])

# Scale y_train
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

# Define the kernel
kernel = C(1.0, (1e-10, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=0.1) + WhiteKernel(noise_level=1)

# Fit the Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_train, y_train_scaled)

# Print the learned kernel and log-marginal likelihood
print(f"Kernel: {gp.kernel_}")
print(f"Log-Marginal-Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta)}")

# Predict
X_test = np.linspace(0, 0.2, 100).reshape(-1, 1)
y_pred_scaled, y_std = gp.predict(X_test, return_std=True)

# Inverse transform the predictions
y_pred = scaler.inverse_transform(y_pred_scaled)

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.plot(X_test, y_pred, color='blue', label='GPR Prediction')
plt.fill_between(X_test.flatten(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, color='blue', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Parameter')
plt.ylabel('Output')
plt.title('Gaussian Process Regression with Composite Kernel')
plt.legend()
plt.grid(True)
plt.show()
