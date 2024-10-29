import numpy as np

N = 100

Theta_min, Theta_max = 0, 10

Step = (Theta_max-Theta_min)/N

K = 1

Phi = np.linspace(0, 0, N)

Theta = np.linspace(Theta_min, Theta_max, N)


def Thidiagonal(A,B,C,F):

    alpha = np.zeros(N)
    beta  = np.zeros(N)

    for i in range(N-1):
        alpha[i+1] = -C[i]/(A[i]*alpha[i]+B[i])
        beta[i+1]  = (F[i]-A[i]*beta[i])/(A[i]*alpha[i]+B[i])

    x = np.zeros(N)

    x[N-1] =(F[N-1] - A[N-1]*beta[N-1])/(beta[N-1]+A[N-1]*alpha[N-1])

    for i in range(N-1,-1,-1):
        x[i-1] = alpha[i]*x[i]+beta[i]
    
    return x

for i in range(10):

    F = np.zeros(N)

    F[0] = Phi[0]
    F[1:(N-1)] = (Phi[0:(N-2)] - 2*Phi[1:(N-1)] + Phi[2:N])/Step**2 + K*np.cos(Phi[1:(N-1)])
    F[N-1] = (Phi[N-1] - Phi[N-2])/Step

    A = np.full(N, 1/Step**2)
    C = np.full(N, 1/Step**2)
    B = np.full(N, -2/Step**2) - K*np.sin(Phi)

    A[0] = 0
    B[0] = 1
    C[0] = 0

    A[N-1] = -1/Step
    B[N-1] = 1/Step
    C[N-1] = 0

    Phi -= Thidiagonal(A,B,C,F)

    print(Phi)


