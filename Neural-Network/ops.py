import numpy as np
import torch
from torch.optim.optimizer import Optimizer

class Base(Optimizer):
    def __init__(self, params, idx, w, agents, lr=0.02, num=0.5, kmult=0.007, name=None, device=None, 
                amplifier=.1, theta=np.inf, damping=.4, eps=1e-5, weight_decay=0, kappa=0.9, stratified=True):

        defaults = dict(idx=idx, lr=lr, w=w, num=num, kmult=kmult, agents=agents, name=name, device=device,
                amplifier=amplifier, theta=theta, damping=damping, eps=eps, weight_decay=weight_decay, kappa=kappa, lamb=lr, stratified=stratified)
        
        super(Base, self).__init__(params, defaults)

    def compute_dif_norms(self, prev_optimizer):
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_dif_norm = 0
            param_dif_norm = 0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None or prev_p.grad is None:
                    continue
                d_p = p.grad.data
                prev_d_p = prev_p.grad.data
                grad_dif_norm += (d_p - prev_d_p).norm().item() ** 2
                param_dif_norm += (p.data - prev_p.data).norm().item() ** 2

            gr, pra = np.sqrt(grad_dif_norm), np.sqrt(param_dif_norm)
            group['grad_dif_norm'] = np.sqrt(grad_dif_norm)
            group['param_dif_norm'] = np.sqrt(param_dif_norm)
        return gr, pra

    def set_norms(self, grad_diff, param_diff):
        for group in self.param_groups:
            group['grad_dif_norm'] = grad_diff
            group['param_dif_norm'] = param_diff
    
    def collect_params(self, lr=False):
        for group in self.param_groups:
            grads = []
            vars = []
            if lr:
                return group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                vars.append(p.data.clone().detach())
                grads.append(p.grad.data.clone().detach())
        return vars, grads
    
    def step(self):
        pass

class CDSGD(Base):
    def __init__(self, *args, **kwargs):
        super(CDSGD, self).__init__(*args, **kwargs)

    def step(self, k, vars=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            num = group['num']
            kmult = group['kmult']
            w = group['w']
            agents = group["agents"]
            device=group["device"]

            lr = num / (kmult * k + 1)
            group["lr"] = lr
            
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                summat = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    summat += w[idx, j] * (vars[j][i+sub].to(device))
                p.data = summat - lr * p.grad.data
        return loss

class CDSGDP(Base):
    def __init__(self, *args, **kwargs):
        super(CDSGDP, self).__init__(*args, **kwargs)

    def step(self, k, vars=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            num = group['num']
            kmult = group['kmult']
            w = group['w']
            agents = group["agents"]
            device=group["device"]
            
            if k == 0:
                group["momentum_p"] = None
            else:
                nest = group["momentum_p"]

            momentum_parm = 0.3
            v_new = []
            lr = num / (kmult * k + 1)
            group["lr"] = lr
            
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                if k == 0:
                    v_t = p.data
                else:
                    v_t = nest[i+sub]
                    
                summat = torch.zeros(p.data.size()).to(device)

                for j in range(agents):
                    summat += w[idx, j] * (vars[j][i+sub].to(device))
                        
                v_t_new = momentum_parm * v_t - lr * p.grad.data
                temp = summat + v_t_new
                
                p.data = temp
                v_new.append(v_t_new)
                
            group["momentum_p"] = v_new

        return loss

class CDSGDN(Base):
    def __init__(self, *args, **kwargs):
        super(CDSGDN, self).__init__(*args, **kwargs)

    def step(self, k, vars=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            num = group['num']
            kmult = group['kmult']
            w = group['w']
            agents = group["agents"]
            device=group["device"]
            
            if k == 0:
                group["momentum_p"] = None
            else:
                nest = group["momentum_p"]
                
            momentum_parm = 0.3
            v_new = []
            lr = num / (kmult * k + 1)
            group["lr"] = lr
            
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                if k == 0:
                    v_t_new = torch.tensor(momentum_parm * p.data - lr * (p.grad.data), requires_grad = True)
                    torch.norm(v_t_new).backward()
                else:
                    v_t_new = torch.tensor(momentum_parm * nest[i+sub] - lr * (p.grad.data + momentum_parm * nest[i+sub].grad.data), requires_grad = True)
                    torch.norm(v_t_new).backward()
                    
                summat = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    summat += w[idx, j] * (vars[j][i+sub].to(device))
                p.data = summat + v_t_new

                v_new.append(v_t_new)
                
            group["momentum_p"] = v_new
        return loss

class DAdSGD(Base):
    def __init__(self, *args, **kwargs):
        super(DAdSGD, self).__init__(*args, **kwargs)
    
    def step(self, k, vars=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device=group["device"]
            
            eps = group['eps']
            lr = group['lr']

            if group["stratified"]:
                alpha = 1
                amplifier = 0.02
            else:
                alpha = 1/.9
                amplifier = 0.05
            
            theta = group['theta']
            grad_dif_norm = group['grad_dif_norm']
            param_dif_norm = group['param_dif_norm']
            lr_new = min(lr * np.sqrt(1 + amplifier * theta), alpha * param_dif_norm / grad_dif_norm) + eps
            theta = lr_new / lr
            group['theta'] = theta
            group['lr'] = lr_new

            lr = lr_new

            sub = 0
            for i, p in enumerate(group['params']):
                summat = torch.zeros(p.data.size()).to(device)

                if p.grad is None:
                    sub -= 1
                    continue

                for j in range(agents):
                    summat += w[idx, j] * (vars[j][i+sub].to(device))

                p.data = summat - lr * p.grad.data

        return loss

class DLAS(Base):
    def __init__(self, *args, **kwargs):
        super(DLAS, self).__init__(*args, **kwargs)

    def collect_lambda(self):
        for group in self.param_groups:
            return group["lamb"]
            
    def step(self, k, vars=None, closure=None, lambdas=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device=group["device"]
            eps = group['eps']
            lr = group['lr']
            
            if group["stratified"]:
                alpha = 1
                amplifier = 0.02
            else:
                alpha = 1/.9
                amplifier = .05

            theta = group['theta']
            grad_dif_norm = group['grad_dif_norm']
            param_dif_norm = group['param_dif_norm']
            kappa = group["kappa"]
            
            if k == 0:
                lamb = lr
                lr_new = lamb
                
            elif k == 1:
                L = grad_dif_norm / param_dif_norm
                lr_new = min(lr * np.sqrt(1 + amplifier * theta), alpha / L ) + eps
                group["L"] = L
                sum_lamb = 0
                for i in range(agents):
                    sum_lamb += w[idx, i] * lambdas[i]
                    
                lamb = sum_lamb + lr_new - lr
                theta = lr_new / lr
                group['theta'] = theta
                
            else:
                old_l = group["L"]
                L = grad_dif_norm / param_dif_norm + kappa * old_l
                lr_new =  min(lr * np.sqrt(1 + amplifier * theta), alpha/L) + eps
                group["L"] = L
                theta = lr_new / lr
                group['theta'] = theta
        
                sum_lamb = 0
                for i in range(agents):
                    sum_lamb += w[idx, i] * lambdas[i]
                
                # Strong fluctuations during early iterations of an epoch mitigated
                if lr_new - lr < 0 and sum_lamb < abs(lr_new - lr):
                    lamb = sum_lamb
                else:
                    lamb = sum_lamb + lr_new - lr

            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                summat = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    summat += w[idx, j] * vars[j][i+sub].to(device)

                p.data = summat - lamb * p.grad.data

            group["lamb"] = lamb
            group["lr"] = lr_new

        return loss

class DAMSGrad(Base):
    def __init__(self, *args, **kwargs):
        super(DAMSGrad, self).__init__(*args, **kwargs)

    def collect_u(self):
        for group in self.param_groups:
            return group["u_tilde_5"]
        
    def step(self, k, vars=None, closure=None, u_tilde_5_all=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        b1 = 0.9
        b2 = 0.99
        epsilon = 1e-6
        alpha = 1e-4

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device=group["device"]
            group['lr'] = alpha

            if k > 0:
                old_m = group["m_t"]
                old_v = group["v_t"]
                old_v_hat = group["v_hat"]

            else:
                old_m = [0]*len(group["params"])
                old_v, old_v_hat = [], []

            m_t_list, v_t_list, v_hat_t_list, u_tilde_5_list = [], [], [], []
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                if k == 0:
                    old_v.append(torch.full_like(p.grad.data, epsilon))
                    old_v_hat.append(torch.full_like(p.grad.data, epsilon))

                m_t = b1 * old_m[i+sub] + (1-b1) * p.grad.data
                v_t = b2 * old_v[i+sub] + (1-b2) * torch.square(p.grad.data)
                v_hat = torch.maximum(old_v_hat[i+sub], v_t)

                summat_param = torch.zeros(p.data.size()).to(device)
                summat_u = torch.zeros(p.data.size()).to(device)

                for j in range(agents):
                    summat_param += w[idx, j] * vars[j][i+sub].to(device)
                    if k == 0:
                        summat_u += w[idx, j] * torch.full_like(v_t, epsilon)
                    else:
                        summat_u += w[idx, j] * u_tilde_5_all[j][i+sub].to(device)

                u_t = torch.maximum(summat_u, torch.full_like(summat_u, epsilon))
                p.data = summat_param - alpha * torch.div(m_t, torch.sqrt(u_t))

                u_tilde_5_list.append((summat_u - old_v_hat[i+sub] + v_hat).clone().detach())
                m_t_list.append(m_t.clone().detach())
                v_t_list.append(v_t.clone().detach())
                v_hat_t_list.append(v_hat.clone().detach())
            
            group["m_t"] = m_t_list
            group["v_t"] = v_t_list
            group["v_hat"] = v_hat_t_list
            group["u_tilde_5"] = u_tilde_5_list
        return loss

class DAdaGrad(Base):
    def __init__(self, *args, **kwargs):
        super(DAdaGrad, self).__init__(*args, **kwargs)

    def collect_u(self):
        for group in self.param_groups:
            return group["u_tilde_5"]
        
    def step(self, k, vars=None, closure=None, u_tilde_5_all=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        b1 = 0.9
        epsilon = 1e-6
        alpha = 1e-4

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device=group["device"]
            group["lr"] = alpha

            if k > 0:
                old_m = group["m_t"]
                old_v_hat = group["v_hat"]

            else:
                old_m = [0]*len(group["params"])
                old_v_hat = []

            m_t_list, v_hat_t_list, u_tilde_5_list = [], [], []
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                if k == 0:
                    old_v_hat.append(torch.full_like(p.grad.data, epsilon))

                m_t = b1 * old_m[i+sub] + (1-b1) * p.grad.data
                v_hat = (k-1)/(k+1)* old_v_hat[i+sub] + (1/(k+1)) * torch.square(p.grad.data)

                summat_param = torch.zeros(p.data.size()).to(device)
                summat_u = torch.zeros(p.data.size()).to(device)

                for j in range(agents):
                    summat_param += w[idx, j] * vars[j][i+sub].to(device)
                    if k == 0:
                        summat_u += w[idx, j] * torch.full_like(v_hat, epsilon)
                    else:
                        summat_u += w[idx, j] * u_tilde_5_all[j][i+sub].to(device)

                u_t = torch.maximum(summat_u, torch.full_like(summat_u, epsilon))
                p.data = summat_param - alpha * torch.div(m_t, torch.sqrt(u_t))

                u_tilde_5_list.append((summat_u - old_v_hat[i+sub] + v_hat).clone().detach())
                m_t_list.append(m_t.clone().detach())
                v_hat_t_list.append(v_hat.clone().detach())
            
            group["m_t"] = m_t_list
            group["v_hat"] = v_hat_t_list
            group["u_tilde_5"] = u_tilde_5_list
        return loss