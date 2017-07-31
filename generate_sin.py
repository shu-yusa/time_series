import numpy as np

def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)

def sin_with_noise(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

if __name__ == "__main__":
    T = 100
    x = np.arange(0, 2 * T + 1)
    sample = sin_with_noise()
    for i in range(24):
        print(sample[i])
    for i in range(24, 200):
        print(sin(x[i]))
