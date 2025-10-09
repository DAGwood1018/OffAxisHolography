import numpy as np


def compute_energy(phase):
    grad_x = np.diff(phase, axis=1, append=phase[:, -1:])
    grad_y = np.diff(phase, axis=0, append=phase[-1:, :])
    return np.sum(grad_x ** 2 + grad_y ** 2)


def reverse_annealing(unwrapped_phase, max_iter=None, initial_temp=100, heat_to_temp=10e6, cool_to_temp=10e-3, heating_rate=1.9, cooling_rate=0.9):
    assert heating_rate > 1, "Heating rate must be greater than 1"
    assert cooling_rate < 1, "Cooling rate must be lower than 1"
    phase = unwrapped_phase.copy()
    T = initial_temp
    count = 0

    # Heat
    while T < heat_to_temp:
        i = np.random.randint(0, phase.shape[0])
        j = np.random.randint(0, phase.shape[1])
        sign = np.random.randint(0, 1)

        E = compute_energy(phase)
        dE = compute_energy(phase[i, j] + 2 * np.pi) - E if sign < 0.5 else compute_energy(phase[i, j] - 2 * np.pi)

        if dE > 0 or np.random.randint(0, 1) > np.exp(-dE / T):
            phase = phase[i, j] + 2 * np.pi if sign < 0.5 else phase[i, j] - 2 * np.pi
        count += 1
        T *= heating_rate

        if max_iter:
            if count == max_iter:
                break

    # Cool
    count = 0
    while T > cool_to_temp:
        i = np.random.randint(0, phase.shape[0])
        j = np.random.randint(0, phase.shape[1])
        sign = np.random.randint(0, 1)

        E = compute_energy(phase)
        dE = compute_energy(phase[i, j] + 2*np.pi) - E if sign < 0.5 else compute_energy(phase[i, j] - 2*np.pi)

        if dE < 0 or np.random.randint(0, 1) < np.exp(-dE / T):
            phase = phase[i, j] + 2*np.pi if sign < 0.5 else phase[i, j] - 2*np.pi
        count += 1
        T *= cooling_rate

        if max_iter:
            if count == max_iter:
                break

    return phase
