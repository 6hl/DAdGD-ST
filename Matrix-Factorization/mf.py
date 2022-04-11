import argparse
import os

import numpy as np

from mfutil import *

def parse_args():
    ''' Function parses command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", default=5, type=int)      
    parser.add_argument("-k", "--kappa", default=0.9, type=float)
    parser.add_argument("-s", "--stratified", default=0, type=int)
    return parser.parse_args()


args = parse_args()

np.random.seed(0)
agent_matrix = np.array([[0.6, 0, 0, 0.4, 0],[0.2, 0.8, 0, 0, 0], [0.2, 0.1, 0.4, 0, 0.3], [0, 0, 0, 0.6, 0.4],[0, 0.1, 0.6, 0, 0.3]])
agents = 5

strat = True if args.stratified == 1 else False
A = load_data(strat)

row, col = A[0].shape
r = 10

param_list = {}
eta_list = {}
grad_list = {}

for i in range(agents):
    param_list[i] = [np.random.randn(row + col, r)]
    eta_list[i] = [1e-9]
    grad_list[i] = []

skip = 200

if strat:
    adaL = 5e4
    iterations = 40000
else:
    adaL = 1e4
    iterations = 100000

L = 1e5
n_L = 1e5

kap = args.kappa

type_t = "homogeneous" if strat else "heterogeneous"
# filename = f"./{type_t}_it{iterations}.csv"

cwd = os.getcwd()
results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

filename = os.path.join(results_path, f"{type_t}_it{iterations}.csv")
print(filename)

train_set = args.test_num

print(f"Training for {iterations}, for test {train_set}, Kappa {kap}")
if train_set == 1:
    CDGD(filename, agent_matrix, A, cgrad, param_list.copy(), L, grad_list.copy(), agents=5, iterations=iterations, skip=skip)
elif train_set == 2:
    CDGD_P(filename, agent_matrix, A, cgrad, param_list.copy(), L, grad_list.copy(), agents=5, iterations=iterations, momentum=0.3, skip=skip)
elif train_set == 3:
    CDGD_N(filename, agent_matrix, A, cgrad, param_list.copy(), n_L, grad_list.copy(), agents=5, iterations=iterations, momentum=0.3, skip=skip)
elif train_set == 4:
    DAdGD(filename, agent_matrix, A,cgrad, param_list.copy(), eta_list.copy(), grad_list.copy(), agents=5, iterations=iterations, skip=skip)
elif train_set == 5:
    DOAS(filename, agent_matrix, A, cgrad, param_list.copy(), eta_list.copy(), grad_list.copy(), agents=5, iterations=iterations, kap=kap, skip=skip)
elif train_set == 6:
    DAMSGrad(filename, agent_matrix, A, cgrad, param_list.copy(), grad_list.copy(), agents=5, iterations=iterations, L=adaL, skip=skip, strat=strat)
elif train_set == 7:
    DAdaGrad(filename, agent_matrix, A, cgrad, param_list.copy(), grad_list.copy(), agents=5, iterations=iterations, L=adaL, skip=skip, strat=strat)