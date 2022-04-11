import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

import csv
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file

from loss_funcs import logistic_loss, logistic_gradient

class Trainer:
    def __init__(
            self,
            data,
            agent_matrix,
            iterations=10000,
            min_allow=0):

        self.iterations_tot = iterations
        self.min_allow = min_allow
        self.agent_matrix = agent_matrix
        self.agents = agent_matrix.shape[0]
        self.eta_time = {}
        self.grad_norm = []

        self.agent_samples = data[0]
        self.agent_targets = data[1]
        self.agent_L = data[2]
        self.agent_L2 = data[3]
        self.agent_parameters = {}
        for i in range(self.agents):
            self.agent_parameters[i] = [np.zeros(self.agent_samples[i].shape[1])]
            self.eta_time[i] = []

    def train(self):
        self.losses = []
        self.iterations = []
        self.grads = {}
        for i in range(self.agents):
            self.grads[i] = []
        
        self.init_run()
        self.compute_loss(iteration=0)
        for i in range(self.iterations_tot):
            for j in range(self.agents):
                tol = self.compute_grad(agent_num=j, iteration=i)

            if not tol:
                break
            self.step(iteration=i)
            if i % 1 == 0 and i != 0:
                self.compute_loss(iteration=i)

    def compute_loss(self, iteration):
        temp_loss = 0
        for i in range(self.agents):
            temp_loss += logistic_loss(self.agent_parameters[i][iteration], self.agent_samples[i], self.agent_targets[i], self.agent_L2[i])

        self.losses.append(temp_loss/self.agents)
        self.iterations.append(iteration)
        for i in range(self.agents):
            if self.name == "DOAS":
                self.lambd_time[i].append(self.lambda_k[i][iteration])
                self.eta_time[i].append(self.eta[i][iteration])
            else:
                self.eta_time[i].append(self.eta[i][iteration])

        if iteration > 0:
            self.grad_norm.append(np.linalg.norm(sum([grad[-1] for grad in self.grads.values()])/self.agents))
            # print(self.grad_norm[iteration-1])

        et = sum([et[-1] for et in self.eta.values()])/self.agents

        if iteration % 200 == 0:
            if self.name == "DOAS":
                temp2 = sum([lamb[-1] for lamb in self.lambda_k.values()])/self.agents
                print(f"Optimizer: {self.name}, Iteration: {iteration}, Loss: {temp_loss/self.agents}, Lambda: {(temp2)}, LR: {et}")
            else:
                print(f"Optimizer: {self.name}, Iteration: {iteration}, Loss: {temp_loss/self.agents}, Eta: {et}")

    def compute_grad(self, agent_num,iteration, init=False, save=True):
        if self.name == "CDGDN" and not init:
            params = self.agent_parameters[agent_num][iteration] + self.momentum_param * self.momentum[agent_num]
        elif self.name == "CDGDN" and init:
            params = self.agent_parameters[agent_num][iteration] + self.momentum_param * self.agent_parameters[agent_num][iteration]
        else:    
            params = self.agent_parameters[agent_num][iteration]

        samples = self.agent_samples[agent_num]
        targets = self.agent_targets[agent_num]
        l2 = self.agent_L2[agent_num]
        if not save:
            return logistic_gradient(params, samples, targets, l2)
        else:
            self.grads[agent_num].append(logistic_gradient(params, samples, targets, l2))
        return la.norm(self.grads[agent_num][-1]) > self.min_allow
    
    def init_run(self):
        pass

    def step(self):
        pass

    def plot_lrs(self, jump, start):
        end = -len(self.iterations) + len(self.iterations)//20
        markerstyles = ["*", "s", "P", "8", "^", "D", "*", "s", "P", "8", "^", "D"]
        if self.name != "CDGD" or self.name != "CDGD-P" or self.name != "CDGD-N":
            for k, v in self.eta_time.items():
                plt.plot(self.iterations[start:end:jump], v[start:end:jump], label=f"{self.name} Agent {k}", marker=markerstyles[k], markevery=self.iterations_tot // 200)
        else:
            plt.plot(self.iterations[start:end:jump], self.eta_time[0][start:end:jump], label=f"1/L", marker=markerstyles[5], markevery=self.iterations_tot // 200)
    
    def eta_smooth(self, jump, start, end_div, ax=None):
        # end = -len(self.iterations)//2
        end = -len(self.iterations) + len(self.iterations)//end_div
        markerstyles = ["*", "s", "P", "8", "h", "D","*", "+","o","*", "s", "P", "8", "h", "D","*", "+","o"]
        if self.name != "CDGD" or self.name != "CDGD-P" or self.name != "CDGD-N":
            for k, v in self.eta_time.items():
                ax.plot(self.iterations[start:end:jump], v[start:end:jump], label=f"{self.name} Agent {k}", marker=markerstyles[k], markevery=self.iterations_tot // 200)
        else:
            plt.plot(self.iterations[start:end:jump], self.eta_time[0][start:end:jump], label=f"1/L", marker=markerstyles[5], markevery=self.iterations_tot // 200)
        ax.legend(loc='upper right')
        ax.grid(visible=True)
        ax.set_yscale("log")
        ax.set(xlabel="Iteration", ylabel=r"$\eta_i^k$")

    def eta_lamb_smooth(self, jump, start, end_div, ax=None):
        # end = -len(self.iterations)//2
        end = -len(self.iterations) + len(self.iterations)//end_div
        markerstyles = ["*", "s", "P", "8", "h", "D","*", "+","o","*", "s", "P", "8", "h", "D","*", "+","o"]
        if self.name != "CDGD" or self.name != "CDGD-P" or self.name != "CDGD-N":
            for k, v in self.eta_time.items():
                ax.plot(self.iterations[start:end:jump], v[start:end:jump], label=f"{self.name} Agent {k}", marker=markerstyles[k], markevery=self.iterations_tot // 200)
        else:
            plt.plot(self.iterations[start:end:jump], self.eta_time[0][start:end:jump], label=f"1/L", marker=markerstyles[5], markevery=self.iterations_tot // 200)
        ax.legend(loc='upper right')
        ax.grid(visible=True)
        ax.set_yscale("log")
    
    def plot_lambd_eta(self, jump, start, end_div, ax1, ax2):
        
        end = -len(self.iterations) + len(self.iterations)//end_div
        every_steps = 800
        markerstyles = ["*", "s", "P", "8", "h", "D","*", "+","o","*", "s", "P", "8", "h", "D","*", "+","o"]    
        for k, v in self.eta_time.items():
            ax1.plot(self.iterations[start:end:jump], v[start:end:jump], label=f"Agent {k}", marker=markerstyles[k], markevery=self.iterations_tot // every_steps)
        ax1.legend(loc='upper right')
        ax1.grid(visible=True)

        for k, v in self.lambd_time.items():
            ax2.plot(self.iterations[start:end:jump], v[start:end:jump], label=f"Agent {k}", marker=markerstyles[k], markevery=self.iterations_tot // every_steps)
        ax2.legend(loc='upper right')
        ax2.grid(visible=True)
        ax1.set(ylabel=r"Smoothed $\eta_i^k$")
        ax1.set_yscale("log")
        ax2.set_yscale("log")

        ax2.set(xlabel="Iteration", ylabel=r"$\lambda_i^k$")
    
    def save_data(self, filename):
        with open(filename, mode="a") as csv_file:
            file = csv.writer(csv_file, lineterminator= "\n")
            file.writerow([self.name])
            file.writerow(self.iterations)
            file.writerow(self.final_losses_plotted)
            # file.writerow(self.learning_rates)
            if self.name == "DOAS":
                for i in range(self.agents):
                    file.writerow(self.lambd_time[i])
                for i in range(self.agents):
                    file.writerow(self.eta_time[i])
            else:
                for i in range(self.agents):
                    file.writerow(self.eta_time[i])

    def plotter(self, f_opt, markerstyle, skip):
        #f_opt = 0
        los =[losses - f_opt for losses in self.losses]
        self.final_losses_plotted = los
        mrk = 1 if self.iterations_tot < 10 else 20
        plt.plot(self.iterations[::skip], los[::skip], label=self.name, marker=markerstyle, markevery=self.iterations_tot // mrk)

    def plot_grad_norm(self, markerstyle, skip):
        plt.plot(self.iterations[::skip], self.grad_norm[::skip], label=self.name, marker=markerstyle, markevery=self.iterations_tot // 20)

def plot_eta_lambda_smooth(optimizers, jump, start, alpha):
    end_div = 20
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharey=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharey=ax1)
    ax = [ax1, ax2]
    for i, (opt, a) in enumerate(zip(optimizers, ax)):
        if i == 1:
            opt.plot_lambd_eta(jump, start, end_div=end_div, ax1=ax2, ax2=ax3)
        else:    
            opt.eta_smooth(jump, start, end_div=end_div, ax=a)

    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig(f"stepsize_comparison_a{alpha}.jpg")
    plt.clf()

def plot_all_losses(optimizers, skip):
    fig = plt.figure()
    markerstyles = ["*", "s", "P", "8", "h", "D","*", "+","o","*", "s", "P", "8", "h", "D","*", "+","o"]
    f_opt = np.min([np.min(opt.losses) for opt in optimizers])
    f_opt=0
    for i, opt in enumerate(optimizers):
        opt.plotter(f_opt, markerstyles[i], skip=skip)
    plt.ylabel("F(x)-F*")
    # plt.ylabel("F(x)")
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    #plt.savefig("LinearReg_Loss.jpg")
    plt.clf()

def plot_grad_norm(optimizer, skip):
    fig = plt.figure()
    markerstyles = ["*", "s", "P", "8", "h", "D","*", "+"]
    for i, opt in enumerate(optimizer):
        opt.plot_grad_norm(markerstyles[i], skip=skip)
    plt.ylabel(r"$\left\| \nabla F(x) \right\|$")
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig("Grad_norm.jpg")
    plt.clf()

def load_data(stratified, data_path, agents=5):
    agent_samples, agent_targets = {}, {}
    agent_L, agent_L2 = {}, {}

    data = load_svmlight_file(data_path)
    total_samples, total_targets = data[0].toarray(), data[1]
    if (np.unique(total_targets) == [1, 2]).all():
        total_targets -= 1
    
    total_samples, total_targets = shuffle(total_samples, total_targets)
    leng = int(len(total_samples))
    agent_sample_len, remainder = int(leng//agents), leng % agents

    if stratified == True:
        for i in range(agents):
            agent_samples[i] = total_samples[i*agent_sample_len:i*agent_sample_len+agent_sample_len]
            agent_targets[i] = total_targets[i*agent_sample_len:i*agent_sample_len+agent_sample_len]
            agent_L[i] = logistic_smoothness(agent_samples[i])
            agent_L2[i] = agent_L[i] / (10 * agent_samples[i].shape[0])

    else:
        targets_0 = np.array(list(map(lambda x: x, np.where(total_targets == 0))))[0]
        targets_1 = np.array(list(map(lambda x: x, np.where(total_targets == 1))))[0]

        agent_samples[0] = total_samples[targets_0[:agent_sample_len]]
        agent_targets[0] = total_targets[targets_0[:agent_sample_len]]
        agent_samples[1] = total_samples[targets_0[agent_sample_len+1:agent_sample_len*2+1]]
        agent_targets[1] = total_targets[targets_0[agent_sample_len+1:agent_sample_len*2+1]]
        targets_0 = np.delete(targets_0, [range(1,agent_sample_len*2+1)])
    
        remaining_samples = total_samples[np.append(targets_0, targets_1)]
        remaining_targets = total_targets[np.append(targets_0, targets_1)]

        agent_samples[2] = remaining_samples[:agent_sample_len]
        agent_targets[2] = remaining_targets[:agent_sample_len]
        agent_samples[3] = remaining_samples[agent_sample_len+1:agent_sample_len*2+1]
        agent_targets[3] = remaining_targets[agent_sample_len+1:agent_sample_len*2+1]
        agent_samples[4] = remaining_samples[agent_sample_len*2+1:agent_sample_len*3+1]
        agent_targets[4] = remaining_targets[agent_sample_len*2+1:agent_sample_len*3+1]
        for i in range(agents):
            agent_L[i] = logistic_smoothness(agent_samples[i])
            agent_L2[i] = agent_L[i] / (10 * agent_samples[i].shape[0])

    return (agent_samples, agent_targets, agent_L, agent_L2)
    

def logistic_smoothness(samples, covtype=False):
    if covtype:
        return 0.25 * np.max(la.eigvalsh(samples.T @ samples / (samples.shape[0])))
    else:
        return 0.25 * np.max(la.eigvalsh(samples.T @ samples / (samples.shape[0])))