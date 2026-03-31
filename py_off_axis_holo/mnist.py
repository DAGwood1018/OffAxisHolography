import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# -----------------------------
# Load MNIST
# -----------------------------
(x_train, _), _ = mnist.load_data()
img = x_train[0].astype(np.float32) / 255.0  # normalize


# -----------------------------
# Interferometric forward model
# -----------------------------
def simulate_interferometry(image, sampling_fraction=0.2, seed=42):
    np.random.seed(seed)

    # Fourier transform (visibility function)
    fft = np.fft.fftshift(np.fft.fft2(image))

    # Generate random sampling mask (u-v coverage)
    mask = np.random.rand(*fft.shape) < sampling_fraction

    # Apply mask (simulate sparse baselines)
    sampled_fft = fft * mask

    return fft, sampled_fft, mask


# -----------------------------
# Reconstruction
# -----------------------------
def reconstruct(sampled_fft):
    recon = np.fft.ifft2(np.fft.ifftshift(sampled_fft))
    return np.abs(recon)


# -----------------------------
# Run simulation
# -----------------------------
fft, sampled_fft, mask = simulate_interferometry(img, sampling_fraction=0.1)
recon = reconstruct(sampled_fft)

# -----------------------------
# Plot results
# -----------------------------
fig, axs = plt.subplots(1, 4, figsize=(12, 3))

axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original")

axs[1].imshow(np.log(np.abs(fft) + 1e-6), cmap='inferno')
axs[1].set_title("FFT (Visibilities)")

axs[2].imshow(mask, cmap='gray')
axs[2].set_title("Sampling Mask")

axs[3].imshow(recon, cmap='gray')
axs[3].set_title("Reconstruction")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()