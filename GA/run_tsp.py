# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 22:03:10 2018

@author: 周宝航
"""

from genetic_algorithm import TSPGA
from tsp import TSP

N_CITIES = 20
CROSS_RATE = 0.1
MUTATE_RATE = 0.08
POP_SIZE = 500
N_GENERATIONS = 100

tsp_model = TSP(N_CITIES)
ga = TSPGA(N_CITIES, POP_SIZE, tsp_model,
                 cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE,
                 n_generations=N_GENERATIONS)

ga.evolve()