import numpy as np

mean = np.load("results/scaler_B_mean.npy")
std = np.load("results/scaler_B_std.npy")

print("Scaler Mean:")
print(mean)

print("Scaler Std:")
print(std)

print("Mean Shape:")
print(mean.shape)

print("Std Shape:")
print(std.shape)