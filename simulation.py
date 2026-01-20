import numpy as np
from LightPipes import *
import matplotlib.pyplot as plt

def generate_beam_data(num_samples=500, grid_size=64):
    wavelength = 500 * nm
    size = 10 * mm
    X = []
    y = []

    for _ in range(num_samples):
        # Select a random propagation distance
        z = np.random.uniform(1, 50) * cm
        
        # Simulate a Gaussian beam
        F = Begin(size, wavelength, grid_size)
        F = GaussBeam(F, 1 * mm)
        F = Forvard(z, F)
        
        # Extract beam intensity
        I = Intensity(F)
        X.append(I)
        
        # Target: predict propagation distance z from beam shape
        y.append(z)
        
    return np.array(X), np.array(y)
