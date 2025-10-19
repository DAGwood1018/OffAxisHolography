import numpy as np
from py_off_axis_holo.hologram_filtering import OffAxisFilter
from py_off_axis_holo.holography_helpers import discrete_residue
from py_off_axis_holo.phase_unwrapping import PhaseUnwrapTIE
from py_off_axis_holo.phase_unwrapping import visualize_branch_cuts
from skimage.restoration import unwrap_phase
from matplotlib import pyplot as plt

def plot_frame(plots):
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))

    a = np.max(np.abs(plots[0]))
    p1 = axs[0].imshow(plots[0], vmin=-a, vmax=a, cmap='RdBu')
    fig.colorbar(p1, ax=axs[0])
    axs[0].set_title("Wrapped Phase")

    a = np.max(np.abs(plots[1]))
    p2 = axs[1].imshow(plots[1], vmin=-1, vmax=1, cmap='RdBu')
    fig.colorbar(p2, ax=axs[1])
    axs[1].set_title("Residues")

    plt.tight_layout()
    plt.show()


phase = np.load("unwrap_test.npy")

branch_cut_matrix, T, E = visualize_branch_cuts(phase)

plt.figure()
plt.imshow(branch_cut_matrix, vmin=0, vmax=1)
plt.show()

plt.figure()
plt.plot(T)
plt.show()

plt.figure()
plt.plot(E)
plt.show()






