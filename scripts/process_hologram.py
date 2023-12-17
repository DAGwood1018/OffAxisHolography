from matplotlib import pyplot as plt
from fluid_driver.system_constants import *
from py_offaxis_holo.phase_unwrapping import *
from py_offaxis_holo.field_propagation import *
from off_axis_utils.holo_tools import height_profile

n = 1.5  # index of refraction of water at ~589nm
n0 = 1.000293  # index of refraction of air at ~589nm


N = 50
path = "C:/Users/dagwo/Documents/My Documents/Code/OpticalFluidControl/OFF/shots/8_1_23/USAF_target/"
bckgrnd = 'bckgrnd_'
obj = 'target_0.npy'
bck_field = np.load(path + bckgrnd + "0.npy")
for n in range(1, N):
    bck_field += np.load(path + bckgrnd + str(n) + '.npy')
bck_field /= N

field = np.load(path + obj) / bck_field

Nx, Ny = field.shape
unwrapper = PhaseUnwrap(Nx, Ny, nb=1, zero=True)
phase = unwrapper(np.angle(field))
h = height_profile(phase, wl, n, n0) / 1e-6

plt.figure(1)
plt.subplot(311)
plt.title("Wrapped Phase")
plt.imshow(np.angle(field)/np.pi, cmap='RdBu', vmin=-1, vmax=1)
plt.axis('off')
cbar = plt.colorbar()
cbar.set_label('Phase / Pi')

plt.subplot(312)
plt.title("Unwrapped Phase")
plt.imshow(phase/np.pi, cmap='Blues')
plt.axis('off')
cbar = plt.colorbar()
cbar.set_label('Phase / Pi')

plt.subplot(313)
plt.title("Height")
plt.imshow(h)
plt.axis('off')
cbar = plt.colorbar()
cbar.set_label("Height [um]")
plt.show()

Nx, Ny = field.shape
dists = np.array([225.55])  # in mm
P = AngularSpectrum(Nx, Ny, wl, pxl_sz)
fields = rep_propagation(P, field, dists*1e-3)

for i, dist in enumerate(dists):
    rec = angular_spectrum(field, dist, wl, pxl_sz)

    plt.figure(i)
    plt.subplot(211)
    plt.imshow(np.abs(field), cmap='gray')
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(np.abs(rec), cmap='gray')
    plt.colorbar()
    plt.show()