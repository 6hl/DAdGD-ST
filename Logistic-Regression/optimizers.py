import numpy as np

import numpy.linalg as la

from util import Trainer

class CDGD(Trainer):
    def __init__(self, *args, **kwargs):
        super(CDGD, self).__init__(*args, **kwargs)
        self.name = "CDGD"

    def step(self, iteration):
        summation = {}
        
        for i in range(self.agents):
            self.eta[i].append(1 / self.agent_L[i])
            summation[i] = np.zeros_like(self.agent_parameters[i][iteration])
            for j in range(self.agents):
                summation[i] += self.agent_matrix[i,j] * self.agent_parameters[j][iteration]

        for i in range(self.agents):
            self.agent_parameters[i].append(summation[i] - self.eta[i][iteration] * self.grads[i][iteration])
    
    def init_run(self):
        self.learning_rates = []
        self.eta = {}
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]

class CDGDP(Trainer):
    def __init__(self, momentum_param=0.3, *args, **kwargs):
        super(CDGDP, self).__init__(*args, **kwargs)
        self.momentum_param = momentum_param
        self.name = "CDGD-P"

    def step(self, iteration):
        summation = {}
        for i in range(self.agents):
            self.eta[i].append(1 / self.agent_L[i])
            summation[i] = np.zeros_like(self.agent_parameters[i][iteration])
            for j in range(self.agents):
                summation[i] += self.agent_matrix[i,j] * self.agent_parameters[j][iteration]

        for i in range(self.agents):
            self.momentum[i] = self.momentum_param * self.momentum[i] - self.eta[i][iteration] * self.grads[i][iteration]
            self.agent_parameters[i].append(summation[i] + self.momentum[i])
    
    def init_run(self):
        self.learning_rates = []
        self.momentum = {}
        self.eta = {}
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]
            self.compute_grad(agent_num=i, iteration=0)
            self.momentum[i] = self.momentum_param * self.agent_parameters[i][0] - self.eta[i][0] * self.grads[i][0]
           
class CDGDN(Trainer):
    def __init__(self, momentum_param=0.3, *args, **kwargs):
        super(CDGDN, self).__init__(*args, **kwargs)
        self.momentum_param = momentum_param
        self.name = "CDGD-N"

    def step(self, iteration):
        summation = {}
        for i in range(self.agents):
            self.eta[i].append(1 / self.agent_L[i])
            summation[i] = np.zeros_like(self.agent_parameters[i][iteration])
            for j in range(self.agents):
                summation[i] += self.agent_matrix[i,j] * self.agent_parameters[j][iteration]

        for i in range(self.agents):
            self.momentum[i] = self.momentum_param * self.momentum[i] - self.eta[i][iteration] * self.grads[i][iteration]
            self.agent_parameters[i].append(summation[i] + self.momentum[i])
    
    def init_run(self):
        self.learning_rates = []
        self.momentum = {}
        self.eta = {}
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]
            self.compute_grad(agent_num=i, iteration=0, init=True)
            self.momentum[i] = self.momentum_param * self.agent_parameters[i][0] - self.eta[i][0] * self.grads[i][0]
            
class DAdGD(Trainer): # eta= 1e-10
    def __init__(self, eps=0.0, eta=None, *args, **kwargs):
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        super(DAdGD, self).__init__(*args, **kwargs)
        self.eps = eps
        self.theta = {}

        for j in range(self.agents):
            self.theta[j] = np.inf

        if not eta:
            self.eta = {}
            for i in range(self.agents):
                self.eta[i] = [1e-10]
        
        self.name = "DAdGD"

    def compute_eta(self, j, iteration=None):
        L = la.norm(self.grads[j][iteration] - self.grads[j][iteration-1]) / la.norm(self.agent_parameters[j][iteration] - self.agent_parameters[j][iteration-1])

        if np.isinf(self.theta[j]):
            lr_new = 0.5/L
        else:
            lr_new = min(np.sqrt(1 + self.theta[j]) * self.eta[j][iteration-1], self.eps/self.eta[j][iteration-1] +  0.5 / L)

        self.theta[j] = lr_new / self.eta[j][iteration-1]
        self.eta[j].append(lr_new)
    
    def step(self, iteration):
        summation = {}
        for j in range(self.agents):
            if iteration == 0:
                self.eta[j] = [1e-10]
            else:
                self.compute_eta(j=j, iteration=iteration)

        for i in range(self.agents):
            summation[i] = np.zeros_like(self.agent_parameters[i][iteration])
            for j in range(self.agents):
                summation[i] += self.agent_matrix[i,j] * self.agent_parameters[j][iteration]

        for i in range(self.agents):
            self.agent_parameters[i].append(summation[i] - self.eta[i][iteration] * self.grads[i][iteration])

    def init_run(self):
        self.learning_rates = []

class DAdGDF(DAdGD):
    def __init__(self, kap, *args, **kwargs):
        super(DAdGDF, self).__init__(*args, **kwargs)
        self.name = "DAdGD-F"
        self.L_list = {}
        self.theta_list = {}
        self.kap = kap
        for i in range(self.agents):
            self.L_list[i] = [0]
            self.theta_list[i] = [self.theta[i]]

    def compute_eta(self, j, iteration):
        L = la.norm(self.grads[j][iteration] - self.grads[j][iteration-1]) / la.norm(self.agent_parameters[j][iteration] - self.agent_parameters[j][iteration-1])
        
        if iteration == 1:
            L_tilde = L
            self.L_list[j].append(L_tilde)
        else:
            L_tilde = L + self.kap * self.L_list[j][-1]
            self.L_list[j].append(L_tilde)

        if np.isinf(self.theta[j]):
            lr_new = 0.5/self.L_list[j][-1]
        else:
            lr_new = min(np.sqrt(1 + self.theta[j]) * self.eta[j][iteration-1], self.eps/self.eta[j][iteration-1] + 0.5 / self.L_list[j][-1])
        
        self.theta[j] = lr_new / self.eta[j][iteration-1]
        self.eta[j].append(lr_new)

class DAdGDST(DAdGDF):
    def __init__(self, *args, **kwargs):
        super(DAdGDST, self).__init__(*args, **kwargs)
        self.name = "DAdGD-ST"

    def step(self, iteration):
        summation = {}
        if iteration == 0:
            for i in range(self.agents):
                summation[i] = np.zeros_like(self.agent_parameters[i][iteration])
                for j in range(self.agents):
                    summation[i] += self.agent_matrix[i,j] * self.agent_parameters[j][iteration]

            for i in range(self.agents):
                self.agent_parameters[i].append(summation[i] - self.lambda_k[i][iteration] * self.grads[i][iteration])
                
        else:
            summation_lambda = {}
            for i in range(self.agents):
                self.compute_eta(i, iteration)

            for i in range(self.agents):
                summation[i] = np.zeros_like(self.agent_parameters[i][iteration-1])
                summation_lambda[i] = np.zeros_like(self.lambda_k[i][iteration-1])
                
                for j in range(self.agents):
                    summation[i] += self.agent_matrix[i,j] * self.agent_parameters[j][iteration]
                    summation_lambda[i] += self.agent_matrix[i,j] * self.lambda_k[j][iteration-1]

            for i in range(self.agents):
                self.lambda_k[i].append(summation_lambda[i] + self.eta[i][iteration] - self.eta[i][iteration-1])
                self.agent_parameters[i].append(summation[i] - self.lambda_k[i][iteration] * self.grads[i][iteration])

    def init_run(self):
        start = 1e-10
        self.old_eta = {}
        self.learning_rates = []
        self.summation_lambda = {}
        self.lambda_k = {}
        self.lambd_time = {}
        for i in range(self.agents):
            self.lambda_k[i] = [start]
            self.lambd_time[i] = []
            self.eta[i] = [start]

class DAdGDGT(Trainer):
    def __init__(self, eps=0.0, eta=None, *args, **kwargs):
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        super(DAdGDGT, self).__init__(*args, **kwargs)
        self.eps = eps
        self.theta = {}

        for j in range(self.agents):
            self.theta[j] = np.inf

        if not eta:
            self.eta = {}
            for i in range(self.agents):
                self.eta[i] = [1e-10]
        self.name = "DAdGD-GT"

    def compute_eta(self, j, iteration=None):
        L = la.norm(self.y_list[j][iteration] - self.y_list[j][iteration-1]) / la.norm(self.agent_parameters[j][iteration] - self.agent_parameters[j][iteration-1])

        if np.isinf(self.theta[j]):
            lr_new = 0.5/L
        else:
            lr_new = min(np.sqrt(1 + self.theta[j]) * self.eta[j][iteration-1], self.eps/self.eta[j][iteration-1] +  0.5 / L)

        self.theta[j] = lr_new / self.eta[j][iteration-1]
        self.eta[j].append(lr_new)
    
    def step(self, iteration):
        summation = {}
        summat_y = {}
        for j in range(self.agents):
            if iteration == 0:
                self.eta[j] = [1e-10]
                self.y_list[j] = [self.grads[j][0]]
            else:
                self.compute_eta(j=j, iteration=iteration)

        for i in range(self.agents):
            summation[i] = np.zeros_like(self.agent_parameters[i][iteration])
            summat_y[i] = np.zeros_like(self.grads[i][iteration])

            for j in range(self.agents):
                summation[i] += self.agent_matrix[i,j] * self.agent_parameters[j][iteration]
                summat_y[i] += self.agent_matrix[i,j] * self.y_list[j][iteration]

        for i in range(self.agents):
            self.agent_parameters[i].append(summation[i] - self.eta[i][iteration] * self.y_list[i][-1])
            grad = self.compute_grad(agent_num=i, iteration=iteration+1, save=False)
            self.y_list[i].append(summat_y[i] + grad - self.grads[i][iteration])

    def init_run(self):
        self.y_list = {}

class DAMSGrad(Trainer):
    def __init__(self, eta=1e-3, *args, **kwargs):
        super(DAMSGrad, self).__init__(*args, **kwargs)
        self.b1 = 0.9
        self.b2 = 0.99

        self.eta = {}
        for i in range(self.agents):
            # self.eta[i] = [eta]
            self.eta[i] = [np.exp(.5)/(16*self.agent_L[i])]
        self.name = "DAMSGrad"

    def compute_step_parm(self, j, iteration=None):
        self.m_t[j].append(self.b1 * self.m_t[j][iteration-1] + (1-self.b1) * self.grads[j][iteration])
        self.v_t[j].append(self.b2 * self.v_t[j][iteration-1] + (1-self.b2) * np.power(self.grads[j][iteration], 2))
        temp1 = np.linalg.norm(self.v_hat_t[j][iteration-1])
        temp2 = np.linalg.norm(self.v_t[j][-1])
        arr = self.v_hat_t[j][iteration-1].copy() if temp1 >= temp2 else self.v_t[j][-1].copy()
        self.v_hat_t[j].append(arr)
    
    def step(self, iteration):
        sum_parm = {}
        if iteration == 0:
            for i in range(self.agents):
                self.eta[i].append(np.exp(.5)/(16*self.agent_L[i]))
                self.u_tilde_5[i].append(np.full_like(self.grads[i][0], self.epsilon))
                self.v_t[i].append(np.full_like(self.grads[i][0], self.epsilon))
                self.v_hat_t[i].append(np.full_like(self.grads[i][0], self.epsilon))
                self.m_t[i].append(np.zeros_like(self.grads[i][0]))
                self.agent_parameters[i].append(self.agent_parameters[i][-1].copy())
        
        else:
            for j in range(self.agents):
                self.eta[j].append(np.exp(.5)/(16*self.agent_L[j]))
                self.compute_step_parm(j=j, iteration=iteration)

            for i in range(self.agents):
                sum_parm[i] = np.zeros_like(self.agent_parameters[i][iteration])
                self.u_tilde[i].append(np.zeros_like(self.u_tilde_5[i][iteration-1]))
                for j in range(self.agents):
                    sum_parm[i] += self.agent_matrix[i,j] * self.agent_parameters[j][iteration]
                    self.u_tilde[i][-1] += self.agent_matrix[i,j] * self.u_tilde_5[j][iteration-1]

                temp1 = np.linalg.norm(self.u_tilde[i][-1])
                temp2 = np.linalg.norm(np.full_like(self.u_tilde[i][-1], self.epsilon))
                arr = self.u_tilde[i][-1].copy() if temp1 >= temp2 else np.full_like(self.u_tilde[i][-1], self.epsilon)
                self.u_t[i].append(arr)
            
            for i in range(self.agents):
                self.agent_parameters[i].append(sum_parm[i] - self.eta[i][-1] * self.m_t[i][-1]/np.sqrt(self.u_t[i][-1]))
                self.u_tilde_5[i].append(self.u_tilde[i][-1] - self.v_hat_t[i][-2]+self.v_hat_t[i][-1])

    def init_run(self):
        self.y_list = {}
        self.epsilon = 1e-6
        self.old_eta = {}
        self.learning_rates = []
        self.m_t = {}
        self.v_t = {}
        self.v_hat_t = {}
        self.u_tilde_5 = {}
        self.u_tilde = {}
        self.u_t = {}
        for i in range(self.agents):
            self.m_t[i] = []
            self.v_t[i] = []
            self.v_hat_t[i] = []
            self.u_tilde[i] = []
            self.u_tilde_5[i] = []
            self.u_t[i] = []

class DAdaGrad(Trainer):
    def __init__(self, eta=1e-3, *args, **kwargs):
        super(DAdaGrad, self).__init__(*args, **kwargs)
        self.b1 = 0.9
        self.eta = {}
        for i in range(self.agents):
            # self.eta[i] = [eta]
            self.eta[i] = [np.exp(.5)/(16*self.agent_L[i])]
        self.name = "DAdaGrad"

    def compute_step_parm(self, j, iteration=None):
        self.m_t[j].append(self.b1 * self.m_t[j][iteration-1].copy() + (1-self.b1) * self.grads[j][iteration].copy())
        self.v_hat_t[j].append((iteration-1)/(iteration) * self.v_hat_t[j][iteration-1].copy() + 1/(iteration) * np.power(self.grads[j][iteration].copy(), 2))
    
    def step(self, iteration):
        sum_parm = {}
        if iteration == 0:
            for i in range(self.agents):
                self.eta[i].append(np.exp(.5)/(16*self.agent_L[i]))
                self.u_tilde_5[i].append(np.full_like(self.grads[i][0], self.epsilon))
                self.v_hat_t[i].append(np.full_like(self.grads[i][0], self.epsilon))
                self.m_t[i].append(np.zeros_like(self.grads[i][0]))
                self.agent_parameters[i].append(self.agent_parameters[i][-1].copy())
        
        else:
            for j in range(self.agents):
                self.eta[j].append(np.exp(.5)/(16*self.agent_L[j]))
                self.compute_step_parm(j=j, iteration=iteration)

            for i in range(self.agents):
                sum_parm[i] = np.zeros_like(self.agent_parameters[i][iteration])
                self.u_tilde[i].append(np.zeros_like(self.u_tilde_5[i][iteration-1]))
                for j in range(self.agents):
                    sum_parm[i] += self.agent_matrix[i,j] * self.agent_parameters[j][iteration].copy()
                    self.u_tilde[i][-1] += self.agent_matrix[i,j] * self.u_tilde_5[j][iteration-1].copy()

                # temp1 = np.linalg.norm(self.u_tilde[i][-1])
                # temp2 = np.linalg.norm(np.full_like(self.u_tilde[i][-1], self.epsilon))
                # arr = self.u_tilde[i][-1].copy() if temp1 >= temp2 else np.full_like(self.u_tilde[i][-1], self.epsilon)
                self.u_t[i].append(np.maximum(self.u_tilde[i][-1].copy(), np.full_like(self.u_tilde[i][-1], self.epsilon)))
            
            for i in range(self.agents):
                self.agent_parameters[i].append(sum_parm[i] - self.eta[i][-1] * self.m_t[i][-1]/np.sqrt(self.u_t[i][-1]))
                self.u_tilde_5[i].append(self.u_tilde[i][-1].copy() - self.v_hat_t[i][-2].copy() + self.v_hat_t[i][-1].copy())

    def init_run(self):
        self.epsilon = 1e-6
        self.old_eta = {}
        self.learning_rates = []
        self.m_t = {}
        self.v_hat_t = {}
        self.u_tilde_5 = {}
        self.u_tilde = {}
        self.u_t = {}
        for i in range(self.agents):
            self.m_t[i] = []
            self.v_hat_t[i] = []
            self.u_tilde[i] = []
            self.u_tilde_5[i] = []
            self.u_t[i] = []