from genetic_algorithm import GA
import numpy as np
import matplotlib.pyplot as plt
# to find the maximum of this function
def F(x): return np.sin(10*x)*x + np.cos(2*x)*x

DNA_SIZE = 20
DNA_BOUND = [0,1]
POP_SIZE = 200
CROSS_RATE = 0.1
MUTATION_RATE = 5e-3
N_GENERATIONS = 100
X_BOUND = [0,5]


plt.ion()

model = GA(DNA_SIZE, DNA_BOUND, POP_SIZE, F,
           feature_bound=X_BOUND, n_generations=N_GENERATIONS,
           cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE)

model.evolve(verbose=True)

plt.ioff()
plt.show()