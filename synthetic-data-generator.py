# Import required libraries
import numpy as np
import random

# Define global variables
MIN_WAVENUMBER = 0.1
MAX_WAVENUMBER = 2000

def key_parameters(a=5, b='b'):
    """
    Define key parameters based on input case.
    
    Returns:
    tuple: (min_features, max_features, min_width, max_width)
    """
    # Define cases
    cases = {
        (1, 'a'): (1, 15, 2, 10),
        (1, 'b'): (15, 30, 2, 10),
        (1, 'c'): (30, 50, 2, 10),
        (2, 'a'): (1, 15, 2, 25),
        (2, 'b'): (15, 30, 2, 25),
        (2, 'c'): (30, 50, 2, 25),
        (3, 'a'): (1, 15, 2, 75),
        (3, 'b'): (15, 30, 2, 75),
        (3, 'c'): (30, 50, 2, 75),
        (4, 'a'): (2, 15, 2, 20),
        (4, 'b'): (2, 15, 6, 30),
        (4, 'c'): (2, 10, 2, 30),
        (5, 'a'): (2, 5, 2, 30),
        (5, 'b'): (2, 10, 2, 10)
    }
    
    if (a, b) in cases:
        return cases[(a, b)]
    else:
        print('Case not defined correctly')
        return None

def random_parameters_for_chi3(min_features, max_features, min_width, max_width, I_S, wavenumber_axis, n_points):
    """
    Generate random parameters for chi3 spectrum without NRB.
    
    Returns:
    numpy.ndarray: Matrix of parameters (n_lor, 3) where each row is [amplitude, resonance, linewidth]
    """
    n_lor = np.random.randint(min_features, max_features + 1) # Number of peaks
    
    a = np.random.uniform(0.1, 1, n_lor)  # Amplitudes
    
    min_w, max_w = 50, n_points - 50  # Exclude peaks in first and last 50 spectral points
    w = MIN_WAVENUMBER + np.random.uniform(min_w, max_w, n_lor) * (MAX_WAVENUMBER - MIN_WAVENUMBER) / len(I_S)  # Resonance positions
    
    g = random.choices(
        np.concatenate([np.random.uniform(min_width, 10, n_lor), np.random.uniform(10, max_width, n_lor)]),
        weights=n_lor * [9] + n_lor * [1],
        k=n_lor
    )  # Linewidths
    
    return np.column_stack((a, w, g))

def generate_chi3(params, wavenumber_axis):
    """
    Build the normalized chi3 complex vector.
    
    Args:
    params (numpy.ndarray): Parameters matrix (n_lor, 3)
    wavenumber_axis (numpy.ndarray): Wavenumber axis
    
    Returns:
    tuple: (chi3, max_abs_chi3)
    """
    chi3 = np.sum(params[:, 0] / (-wavenumber_axis[:, np.newaxis] + params[:, 1] - 1j * params[:, 2]), axis=1)
    return chi3, np.max(np.abs(chi3))

def sigmoid(x, c, b):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-(x - c) * b))

def polynomial_NRB(nu):
    """Generate polynomial NRB"""
    coeffs = np.random.uniform([-10, -10, -1, -10, -1], [10, 10, 1, 10, 1])
    p = np.polyval(coeffs, nu)
    return (p - p.min()) / (p.max() - p.min())

def generate_sigmoidal_stokes(nu):
    """Generate normalized sigmoidal Stokes NRB shape"""
    bs = np.random.normal(10, 5, 2)
    cs = np.random.normal([0.2, 0.7], [0.3, 0.3])
    sig1 = sigmoid(nu, cs[0], bs[0])
    sig2 = sigmoid(nu, cs[1], -bs[1])
    return sig1 * sig2

def generate_bCARS(min_features, max_features, min_width, max_width, wavenumber_axis, nu, n_points):
    """
    Generate a CARS spectrum.
    
    Returns:
    tuple: (normalized_cars, chi3.imag, I_S) or (None, None, None) if discarded
    """
    MaxNRB, MinNRB = np.random.uniform(0.5, 0.7), np.random.uniform(0.3, 0.5)
    
    # Choose Stokes intensity profile
    choice = random.choice([1, 2, 3])
    if choice == 1:
        I_S = generate_sigmoidal_stokes(nu)
    elif choice == 2:
        mu, sig = np.random.uniform(0, MAX_WAVENUMBER), np.random.randint(500, 1500)
        I_S = np.abs(np.exp(-np.power(wavenumber_axis - mu, 2.) / (2 * np.power(sig, 2.))))
    else:
        I_S = polynomial_NRB(nu)
    
    I_S /= np.max(I_S)
    
    # Generate NRB
    st1 = (MaxNRB - MinNRB) / (MAX_WAVENUMBER - MIN_WAVENUMBER)
    Chi3_NR = st1 * (wavenumber_axis - MIN_WAVENUMBER) + MaxNRB
    if random.choice([0, 1]) == 0:
        Chi3_NR = np.flip(Chi3_NR)
    
    # Generate noise
    max_noise = np.random.uniform(0.005, 0.05)
    noise = np.random.randn(n_points) * np.random.uniform(0.0002, max_noise)
    
    # Generate resonant part
    params = random_parameters_for_chi3(min_features, max_features, min_width, max_width, I_S, wavenumber_axis, n_points)
    Chi3_R, maxChi3_R = generate_chi3(params, wavenumber_axis)
    Chi3_R /= maxChi3_R
    
    # Combine resonant and non-resonant parts
    Chi3 = Chi3_R + Chi3_NR
    cars = (np.abs(Chi3)**2) * I_S + noise
    max_cars, min_cars = np.max(cars), np.min(cars)
    
    # Filter out weak or noisy features
    to_keep = []
    for param in params:
        Chi3_Rs, _ = generate_chi3(param.reshape(1, -1), wavenumber_axis)
        Chi3_Rs /= maxChi3_R
        Chi3 = Chi3_Rs + Chi3_NR
        cars = (np.abs(Chi3)**2) * I_S + noise
        cars_norm = (cars - min_cars) / (max_cars - min_cars)
        
        wcenter = int(param[1] * n_points / (MAX_WAVENUMBER - MIN_WAVENUMBER))
        wmin = int((param[1] - param[2]) * n_points / (MAX_WAVENUMBER - MIN_WAVENUMBER))
        wmax = int((param[1] + param[2]) * n_points / (MAX_WAVENUMBER - MIN_WAVENUMBER))
        
        dmin, dmax = wcenter - wmin, wmax - wcenter
        
        jump = np.mean(cars_norm[wmin:wcenter]) - np.mean(cars_norm[wcenter:wmax])
        jump_b = np.mean(cars_norm[wmin-dmax-dmin:wmin-dmax]) - np.mean(cars_norm[wmin-dmax:wmin])
        jump_a = np.mean(cars_norm[wmax:wmax+dmin]) - np.mean(cars_norm[wmax+dmin:wmax+dmin+dmax])
        
        jumpcomp = np.sum(np.abs(jump - jump_b) + np.abs(jump - jump_a))
        
        to_keep.append(jumpcomp > 3 * max_noise and cars_norm[wcenter] > 0.1)
    
    new_params = params[to_keep]
    if len(new_params) == 0:
        return None, None, None
    
    Chi3_R_C, _ = generate_chi3(new_params, wavenumber_axis)
    Chi3_R_C /= maxChi3_R
    
    Chi3_C = Chi3_R_C + Chi3_NR
    cars_C = (np.abs(Chi3_C)**2) * I_S + noise
    cars_norm_C = (cars_C - min_cars) / (max_cars - min_cars)
    
    return cars_norm_C, Chi3_R_C.imag, I_S

def generate_batch(min_features, max_features, min_width, max_width, n_points, size=10000):
    """
    Generate a batch of CARS spectra.
    
    Returns:
    tuple: (BCARS, RAMAN, NRB) arrays
    """
    step = (MAX_WAVENUMBER - MIN_WAVENUMBER) / n_points
    wavenumber_axis = np.arange(MIN_WAVENUMBER, MAX_WAVENUMBER, step)
    nu = np.linspace(0, 1, n_points)
    
    BCARS = np.empty((size, n_points))
    RAMAN = np.empty((size, n_points))
    NRB = np.empty((size, n_points))
    
    for i in range(size):
        c, r, n = generate_bCARS(min_features, max_features, min_width, max_width, wavenumber_axis, nu, n_points)
        if c is not None and r is not None and n is not None:
            BCARS[i, :], RAMAN[i, :], NRB[i, :] = c, r, n
    
    return BCARS, RAMAN, NRB

def generate_all_data(min_features, max_features, min_width, max_width, N_train, N_valid, n_points):
    """
    Generate training and validation datasets.
    
    Returns:
    tuple: (BCARS_train, RAMAN_train, NRB_train, BCARS_valid, RAMAN_valid, NRB_valid)
    """
    BCARS_train, RAMAN_train, NRB_train = generate_batch(min_features, max_features, min_width, max_width, n_points, N_train)
    BCARS_valid, RAMAN_valid, NRB_valid = generate_batch(min_features, max_features, min_width, max_width, n_points, N_valid)
    return BCARS_train, RAMAN_train, NRB_train, BCARS_valid, RAMAN_valid, NRB_valid

# Main execution
if __name__ == "__main__":
    # Example usage
    min_features, max_features, min_width, max_width = key_parameters(5, 'b')
    N_train, N_valid, n_points = 1000, 200, 1000  # Example values
    
    BCARS_train, RAMAN_train, NRB_train, BCARS_valid, RAMAN_valid, NRB_valid = generate_all_data(
        min_features, max_features, min_width, max_width, N_train, N_valid, n_points
    )
    
    print(f"Generated {N_train} training samples and {N_valid} validation samples")
    print(f"BCARS shape: {BCARS_train.shape}")
    print(f"RAMAN shape: {RAMAN_train.shape}")
    print(f"NRB shape: {NRB_train.shape}")
