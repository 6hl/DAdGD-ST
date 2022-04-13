import csv

import numpy as np
import scipy.linalg as LA
from scipy.sparse import random as r
import pandas as pd

np.random.seed(0)
def load_data(stratified, agents=5):
    # Data generation same as in original paper
    agent_A = {}
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('./u.data', sep='\t', names=names)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    ratings = np.zeros((n_users, n_items))
    for i in range(agents):
        agent_A[i] = ratings.copy()

    if stratified:
        for row in df.itertuples():
            ratings[row[1]-1, row[2]-1] = row[3]
        for i in range(agents):
            agent_A[i] = ratings
    else:
        for row in df.itertuples():
            agent_A[int(row[3]-1)][row[1]-1, row[2]-1] = row[3]
    return agent_A

def func(param, A):
    m, n = A.shape
    U, V = param[:m], param[m:]
    return 0.5* LA.norm(U @ V.T - A)**2

def cgrad(param, A):
    m, n = A.shape
    U, V = param[:m], param[m:]
    res = U @ V.T - A
    grad_U = res @ V
    grad_V = res.T @ U
    return np.vstack([grad_U, grad_V])


def safe_division(x, y):
    return np.exp(np.log(x) - np.log(y)) if y != 0 else 1e4

def CDGD(filename, w, A, cgrad, param_list, L, grad_list, agents=5, iterations=100, skip=20):
    name = "CDGD"
    eta_list = {}
    for n in range(agents):
        eta_list[n] = []
    grad_norms = []
    iters = []
    for i in range(iterations):
        
        for j in range(agents):
            eta_list[j].append(1/L)
            summation = np.zeros_like(param_list[j][0])
            for k in range(agents):
                summation += w[j,k] * param_list[k][0]

            grad_list[j] = cgrad(param_list[j][0], A[j])
            param_list[j].append(summation - eta_list[j][i] * grad_list[j])
        
        for j in range(agents):
            param_list[j].pop(0)

        if i % skip == 0:
            iters.append(i)
            grad_norms.append(np.linalg.norm(sum([grad for grad in grad_list.values()])/agents))
            print(f"{name}, Iteration: {i}, Gradient Norm: {grad_norms[-1]}, Eta: {eta_list[0][-1]}")
    
    record_results(filename, iters, grad_norms, eta_list, name, skip=skip)

def CDGD_P(filename, w, A, cgrad, param_list, L, grad_list, agents=5, iterations=100, momentum=0.9, skip=20):
    name = "CDGD-P"
    eta_list = {}
    v_k = {}
    for n in range(agents):
        eta_list[n] = []
        v_k[n] = [np.zeros_like(param_list[n][0])]
    iters = []
    grad_norms = []
    for i in range(iterations):
        
        for j in range(agents):
            eta_list[j].append(1/L)
            summation = np.zeros_like(param_list[j][0])
            for k in range(agents):
                summation += w[j,k] * param_list[k][0]

            grad_list[j] = cgrad(param_list[j][0], A[j])
            v_k[j].append(momentum * v_k[j][0] - eta_list[j][i] * grad_list[j])
            
            param_list[j].append(summation + v_k[j][-1])

        for j in range(agents):
            param_list[j].pop(0)
            v_k[j].pop(0)

        if i % skip == 0:
            iters.append(i)
            grad_norms.append(np.linalg.norm(sum([grad for grad in grad_list.values()])/agents))
            print(f"{name}, Iteration: {i}, Gradient Norm: {grad_norms[-1]}, Eta: {eta_list[0][-1]}")
    record_results(filename, iters, grad_norms, eta_list, name, skip=skip)

def CDGD_N(filename, w, A, cgrad, param_list, L, grad_list, agents=5, iterations=100, momentum=0.9, skip=20):
    name = "CDGD-N"
    eta_list = {}
    v_k = {}
    for n in range(agents):
        eta_list[n] = []
        v_k[n] = [np.zeros_like(param_list[n][0])]

    grad_norms = []
    iters = []
    for i in range(iterations):
        
        for j in range(agents):
            eta_list[j].append(1/L)
            summation = np.zeros_like(param_list[j][0])
            for k in range(agents):
                summation += w[j,k] * param_list[k][0]

            grad_list[j] = cgrad(param_list[j][0] + momentum * v_k[j][0], A[j])
            v_k[j].append(momentum * v_k[j][0] - eta_list[j][i] * grad_list[j])
            param_list[j].append(summation + v_k[j][-1])

        for j in range(agents):
            param_list[j].pop(0)
            v_k[j].pop(0)

        
        if i % skip == 0:
            iters.append(i)
            grad_norms.append(np.linalg.norm(sum([grad for grad in grad_list.values()])/agents))
            print(f"{name}, Iteration: {i}, Gradient Norm: {grad_norms[-1]}, Eta: {eta_list[0][-1]}")
    
    record_results(filename, iters, grad_norms, eta_list, name, skip=skip)



def DAMSGrad(filename, w, A, cgrad, param_list, grad_list, agents=5, iterations=100, L=1e-5, skip=20, strat=True):
    name = "DAMSGrad"
    eta_list = {}
    m_list = {}
    v_list = {}
    v_hat_list = {}
    u_list = {}
    u_tilde_list = {}
    summation = {}
    u_tilde_list_5 = {}
    if strat:
        b_1 = 0.9
        b_2 = 0.99
        epsilon = 1e-6
    else:
        b_1 = 0.3
        b_2 = 0.99
        epsilon = 1e-4
    temp = cgrad(param_list[0][0], A[0])

    for n in range(agents):
        eta_list[n] = []
        m_list[n] = []
        v_list[n] = []
        v_hat_list[n] = [np.full_like(temp, epsilon)]
        u_list[n] = []
        u_tilde_list[n] = []
        summation[n] = []
        u_tilde_list_5[n] = [np.full_like(temp, epsilon)]

    grad_norms = []
    iters = []

    for i in range(iterations):
        
        for j in range(agents):
            summation_t = np.zeros_like(param_list[j][0])
            summation_u = np.zeros_like(u_tilde_list_5[j][0])
            for k in range(agents):
                summation_t += w[j,k] * param_list[k][-1]
                summation_u += w[j,k] * u_tilde_list_5[k][-1]
            summation[j].append(summation_t)
            u_tilde_list[j].append(summation_u)

        for j in range(agents):
            eta_list[j].append(np.exp(0.5)/(16*L))
            grad_list[j].append(cgrad(param_list[j][-1], A[j]))
            if i == 0:
                m_list[j].append(np.zeros_like(grad_list[j][0]))
                v_list[j].append(np.full_like(grad_list[j][0], epsilon))
                u_tilde_list_5[i].append(np.full_like(grad_list[j][0], epsilon))
                v_hat_list[i].append(np.full_like(grad_list[j][0], epsilon))
                param_list[j].append(param_list[j][-1].copy())
            else:
                m_list[j].append(b_1 * m_list[j][-1] + (1-b_1) * grad_list[j][-1])
                v_list[j].append(b_2 * v_list[j][-1] + (1-b_2) * np.power(grad_list[j][-1],2))
                v_hat_list[j].append(np.maximum(v_hat_list[j][-1].copy(), v_list[j][-1].copy()))
                u_list[j].append(np.maximum(u_tilde_list[j][-1].copy(), epsilon))
                param_list[j].append(summation[j][-1] - eta_list[j][i] * m_list[j][-1]/np.sqrt(u_list[j][-1]))
                u_tilde_list_5[j].append(u_tilde_list[j][-1].copy() - v_hat_list[j][-2].copy() + v_hat_list[j][-1].copy())

                m_list[j].pop(0)
                v_list[j].pop(0)
                v_hat_list[j].pop(0)
                u_list[j].pop(0)
                param_list[j].pop(0)
                u_tilde_list_5[j].pop(0)
                u_tilde_list[j].pop(0)
                grad_list[j].pop(0)
                summation[j].pop(0)
        if i % skip == 0:
            iters.append(i)
            grad_norms.append(np.linalg.norm(sum([grad[-1] for grad in grad_list.values()])/agents))
            print(f"{name}, Iteration: {i}, Gradient Norm: {grad_norms[-1]:.4f}, Eta: {eta_list[0][-1]}")    
    record_results(filename, iters, grad_norms, eta_list, name, skip=skip)

def DAdaGrad(filename, w, A, cgrad, param_list, grad_list, agents=5, iterations=100, L=1e-5, skip=20, strat=True):
    name = "DAdaGrad"
    eta_list = {}
    m_list = {}
    v_hat_list = {}
    u_list = {}
    u_tilde_list = {}
    summation = {}
    u_tilde_list_5 = {}
    if strat:
        b_1 = 0.3
        epsilon = 1e-6
    else:
        b_1 = 0.9
        epsilon = 1e-6
    temp = cgrad(param_list[0][0], A[0])

    for n in range(agents):
        eta_list[n] = []
        m_list[n] = []
        v_hat_list[n] = [np.full_like(temp, epsilon)]
        u_list[n] = []
        u_tilde_list[n] = []
        summation[n] = []
        u_tilde_list_5[n] = [np.full_like(temp, epsilon)]

    grad_norms = []
    iters = []

    for i in range(iterations):
        
        for j in range(agents):
            summation_t = np.zeros_like(param_list[j][0])
            summation_u = np.zeros_like(u_tilde_list_5[j][0])
            for k in range(agents):
                summation_t += w[j,k] * param_list[k][-1]
                summation_u += w[j,k] * u_tilde_list_5[k][-1]
            summation[j].append(summation_t)
            u_tilde_list[j].append(summation_u)

        for j in range(agents):
            eta_list[j].append(np.exp(0.5)/(16*L))
            grad_list[j].append(cgrad(param_list[j][-1], A[j]))
            if i == 0:
                m_list[j].append(b_1 * np.zeros_like(grad_list[j][-1]) + (1-b_1) * grad_list[j][-1])
                v_hat_list[j].append(0 * np.full_like(grad_list[j][-1], epsilon) + (1-0) * np.power(grad_list[j][-1],2))
            else:
                m_list[j].append(b_1 * m_list[j][-1] + (1-b_1) * grad_list[j][-1])
                v_hat_list[j].append((i)/(i+1) * v_hat_list[j][-1] + (1/i) * np.power(grad_list[j][-1],2))

            u_list[j].append(np.maximum(u_tilde_list[j][-1].copy(), np.full_like(u_tilde_list[j][-1], epsilon)))
            param_list[j].append(summation[j][-1] - eta_list[j][i] * m_list[j][-1]/np.sqrt(u_list[j][-1]))
            u_tilde_list_5[j].append(u_tilde_list[j][-1] - v_hat_list[j][-2] + v_hat_list[j][-1])
        
        if i > 0:
            for j in range(agents):
                m_list[j].pop(0)
                v_hat_list[j].pop(0)
                u_list[j].pop(0)
                param_list[j].pop(0)
                u_tilde_list_5[j].pop(0)
                grad_list[j].pop(0)
                summation[j].pop(0)
                u_tilde_list[j].pop(0)

        if i % skip == 0:
            iters.append(i)
            grad_norms.append(np.linalg.norm(sum([grad[-1] for grad in grad_list.values()])/agents))
            print(f"{name}, Iteration: {i}, Gradient Norm: {grad_norms[-1]:.4f}, Eta: {eta_list[0][-1]}")
    
    record_results(filename, iters, grad_norms, eta_list, name, skip=skip) 

def DAdGD(filename, w, A, cgrad, param_list, eta_list, grad_list, agents=5, iterations=100, skip=20):
    iters = []
    theta_list = {}
    i_results = {}
    name = "DAdGD"
    for i in range(agents):
        theta_list[i] = []
        i_results[i] = []

    grad_norms = []
    for i in range(iterations):
        
        if i == 0:
            for j in range(agents):
                grad_list[j].append(cgrad(param_list[j][0], A[j]))
                summation = np.zeros_like(param_list[j][0])
                theta_list[j] = 1e9
                for k in range(agents):
                    summation += w[j, k] * param_list[k][0]
                param_list[j].append(summation - eta_list[j][0] * grad_list[j][0])
            
        else:
            for j in range(agents):
                grad_list[j].append(cgrad(param_list[j][1], A[j]))

                summation = np.zeros_like(param_list[j][1])
                for k in range(agents):
                    summation += w[j, k] * param_list[k][1]

                norm_param = LA.norm(param_list[j][1] - param_list[j][0])
                norm_grads = LA.norm(grad_list[j][1] - grad_list[j][0])
                eta_list[j].append(min(np.sqrt(1 + theta_list[j]) * eta_list[j][i-1],  0.5 / (norm_grads/norm_param)))
                
                param_list[j].append(summation - eta_list[j][i] * grad_list[j][1])
                theta_list[j] = eta_list[j][i]/eta_list[j][i-1]
        
            for j in range(agents):
                param_list[j].pop(0)
                grad_list[j].pop(0)

        if i % skip == 0:
            iters.append(i)
            grad_norms.append(np.linalg.norm(sum([grad[-1] for grad in grad_list.values()])/agents))
            print(f"{name}, Iteration: {i}, Gradient Norm: {grad_norms[-1]}, Eta: {eta_list[0][-1]}")
    record_results(filename, iters, grad_norms, eta_list, name, skip=skip)
    
def DOAS(filename, w, A, cgrad, param_list, eta_list, grad_list, agents=5, iterations=100, kap=0.99, skip=20):
    name = "DOAS"
    iters = []
    theta_list = {}
    i_results = {}
    L_list = {}
    lambda_list = {}
    lambda_sum = {}
    for i in range(agents):
        theta_list[i] = 1e5
        i_results[i] = []
        lambda_list[i] = [eta_list[i][0]]
    grad_norms = []
    for i in range(iterations):            
        lambda_sum = {}
        
        if i == 0:
            for j in range(agents):
                grad_list[j].append(cgrad(param_list[j][0], A[j]))
                summation = np.zeros_like(param_list[j][0])
                for k in range(agents):
                    summation += w[j, k] * param_list[k][0]
                param_list[j].append(summation - lambda_list[j][0] * grad_list[j][0])
        
        else:
            for j in range(agents):
                lambda_sum[j] = np.zeros_like(lambda_list[j][-1])
                for k in range(agents):
                    lambda_sum[j] += w[j, k] * lambda_list[k][-1]

            for j in range(agents):
                grad_list[j].append(cgrad(param_list[j][1], A[j]))
                summation = np.zeros_like(param_list[j][1])

                for k in range(agents):
                    summation += w[j, k] * param_list[k][1]

                norm_param = LA.norm(param_list[j][1] - param_list[j][0])
                norm_grads = LA.norm(grad_list[j][1] - grad_list[j][0])

                if i == 1:
                    L_list[j] = norm_grads / norm_param
                else:
                    L_list[j] = norm_grads/norm_param + kap * L_list[j]
                
                eta_list[j].append(min(np.sqrt(1 + theta_list[j]) * eta_list[j][i-1],  0.5 / L_list[j]))

                if eta_list[j][i] - eta_list[j][i-1] < 0 and lambda_sum[j] <= abs(eta_list[j][i] - eta_list[j][i-1]):
                    lambda_list[j].append(1e-8)
                else:
                    lambda_list[j].append(lambda_sum[j] + eta_list[j][i] - eta_list[j][i-1])

                param_list[j].append(summation - lambda_list[j][1] * grad_list[j][1])
                theta_list[j] = eta_list[j][i]/eta_list[j][i-1]

            for j in range(agents):
                param_list[j].pop(0)
                grad_list[j].pop(0)
                lambda_list[j].pop(0)

        if i % skip == 0:
            iters.append(i)
            grad_norms.append(np.linalg.norm(sum([grad[-1] for grad in grad_list.values()])/agents))
            print(f"{name}, Iteration: {i}, Gradient Norm: {grad_norms[-1]}, Eta: {sum([eta[-1] for eta in eta_list.values()])/agents:.8f} "+
                  f"Lamb: {sum([lamb[-1] for lamb in lambda_list.values()])/agents:.8f}")

    record_results(filename, iters, grad_norms, eta_list, name+str(kap), lambda_list=lambda_list, skip=skip)

def record_results(filename, iters, grad_norm, eta_list, name, lambda_list=None, skip=20):
    agents = 5
    with open(filename, mode="a") as csv_file:
        file = csv.writer(csv_file, lineterminator= "\n")
        file.writerow([name])
        file.writerow(iters)
        file.writerow(grad_norm)
        if name == "DOAS":
            for i in range(agents):
                file.writerow(lambda_list[i][::skip])
            for i in range(agents):
                file.writerow(eta_list[i][::skip])
        else:
            for i in range(agents):
                file.writerow(eta_list[i][::skip])

if __name__ == "__main__":
    grad_norms = []
    grad_list = {}
    for i in range(5):
        grad_list[i] = []
    
    for i in range(5):
        for j in range(3):
            grad_list[i].append(j+i)
        print(grad_list[i])

    print(sum([grad[-2] for grad in grad_list.values()])/5)
