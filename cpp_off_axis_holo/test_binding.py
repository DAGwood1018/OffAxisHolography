import dft_lib
import numpy as np

N=5
shape = np.array([N,N])
test = np.zeros((N,N), dtype=np.complex128)
for i in range(N):
    for j in range(N):
        test[i,j] = np.cos(6*np.pi*i/N + 6*np.pi*j/N)
print(test)
dft = dft_lib.DFT(shape, 1, 0, True)  # Adjust parameters as needed
print("Size:", dft.size())

dft.forwards(test)

dft.to_string(True)
dft.to_string(False)