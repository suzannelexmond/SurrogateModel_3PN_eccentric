import numpy as np

# Generate two vectors A and phi of shape (1000,)
A = np.random.rand(1000)  # Example vector A
phi = np.random.rand(1000)  # Example vector phi

# Reshape A and phi to (1000, 1)
A = A.reshape(-1, 1)
phi = phi.reshape(-1, 1)

# Calculate element-wise multiplication
exp = np.exp(-1j * phi)
y = A * exp

# Print the shapes of A, phi, and y to verify
print(f'Shape of A: {A.shape}')
print(f'Shape of phi: {phi.shape}')
print(f'Shape of y: {y.shape}')

# Print example elements of y for illustration
print('Example elements of y:')
for i in range(5):  # Print first 5 elements
    print(f'y[{i}]: {y[i][0]}, A{i} = {A[i][0]}, exp{i} = {exp[i][0]}')  # Accessing the element since y is of shape (1000, 1)

