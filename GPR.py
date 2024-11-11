import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace your dataset information here
# Load your data in place of these dummy arrays
X = torch.tensor(self.parameter_space_output[:, np.newaxis], dtype=torch.float32, device=device)
X_train = torch.tensor(self.parameter_space_input[self.greedy_parameters_idx].reshape(-1, 1), dtype=torch.float32, device=device)
y_train = torch.tensor(np.squeeze(training_set.T[time_node]), dtype=torch.float32, device=device)

# Step 1: Scale y_train using StandardScaler
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)

# Step 2: Define the GPR model with the specified kernel
class MaternKernelWithScaling(gpytorch.kernels.MaternKernel):
    def __init__(self, nu=1.5):
        super(MaternKernelWithScaling, self).__init__(nu=nu)
        # Constrain length_scale within specified bounds
        self.lengthscale_constraint = gpytorch.constraints.Interval(0.0001, 0.3)

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            MaternKernelWithScaling(nu=1.5)
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Step 3: Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = GPRegressionModel(X_train, y_train_scaled, likelihood).to(device)

# Step 4: Define the training function and optimizer
def train_model(model, likelihood, X_train, y_train_scaled, training_iterations=50):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Optimizer
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train_scaled)
        loss.backward()
        optimizer.step()
        
        # Print the optimized kernel hyperparameters during training
        if i % 10 == 0:
            print(f"Iteration {i + 1}/{training_iterations}, Loss: {loss.item()}")
            print(f"Kernel: {model.covar_module.base_kernel} | Lengthscale: {model.covar_module.base_kernel.lengthscale.item()}")

# Step 5: Train the model
train_model(model, likelihood, X_train, y_train_scaled)

# Step 6: Evaluate the model on test data and visualize the results

model.eval()
likelihood.eval()

with torch.no_grad():
    # Make predictions and scale back to original units
    posterior_distribution = likelihood(model(X))
    mean_prediction_scaled = posterior_distribution.mean.cpu().numpy()
    std_prediction_scaled = np.sqrt(posterior_distribution.variance.cpu().numpy())
    
    # Inverse transform to original scale
    mean_prediction = scaler.inverse_transform(mean_prediction_scaled.reshape(-1, 1)).flatten()
    std_prediction = std_prediction_scaled * scaler.scale_[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(X.cpu().numpy(), mean_prediction, 'b', label='Posterior Mean')
    plt.fill_between(X.cpu().numpy().flatten(), mean_prediction - std_prediction, mean_prediction + std_prediction, color='blue', alpha=0.2, label='Posterior Std. Dev.')
    plt.scatter(X_train.cpu().numpy(), scaler.inverse_transform(y_train_scaled.cpu().numpy().reshape(-1, 1)), color='red', label='Training Data')
    plt.title("Posterior Distribution with GPR")
    plt.legend()
    plt.show()

