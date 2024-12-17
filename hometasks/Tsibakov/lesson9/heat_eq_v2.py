"""
 Tsybakov Mikhail 626 
 15.12.2024
"""



import numpy as np
import matplotlib.pyplot as plt



Lx, Ly = 3.0, 3.0  # Domain size
nx, ny = 128, 128  
dx, dy = Lx / nx, Ly / ny  
alpha  = 0.05         # Thermal diffusivity
t1     = 1.0     
dt     = 0.001  
nt     = int(t1 / dt) 



# Wave vectors
kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
kx, ky = np.meshgrid(kx, ky, indexing="ij")
k_squared = kx**2 + ky**2



# Initial condition
u = np.zeros((nx, ny))
u[nx//2, ny//2] = 100.0



# Fourier transform 
u_hat = np.fft.fft2(u)



# Time integration - implicit spectral scheme
for n in range(nt):
    du_hat_dt = -alpha * k_squared * u_hat

    # Update the solution in Fourier space
    u_hat += dt * du_hat_dt



# Inverse FFT: Spectral --> Physical
u = np.fft.ifft2(u_hat).real



plt.figure(figsize=(8, 6))
plt.imshow(u, extent=[0, Lx, 0, Ly], origin="lower", cmap="jet")
plt.colorbar(label="Temperature")
plt.title("2D Heat Equation with Spectral Method", font = "Consolas", fontsize = 10)
plt.xlabel("x", font = "Consolas", fontsize = 10)
plt.ylabel("y", font = "Consolas", fontsize = 10)
plt.savefig("spectral_heat.png", dpi = 300)
plt.show()