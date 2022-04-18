import argparse
from collections import Counter, OrderedDict
import copy
import csv
import os
from random import shuffle, sample
from time import perf_counter
import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate


from models import *
from ops import DAdSGD, DLAS, CDSGD, CDSGDP, CDSGDN, DAMSGrad, DAdaGrad


warnings.filterwarnings("ignore")

class DTrainer:
    def __init__(self, 
                dataset="cifar10", 
                epochs=100, 
                batch_size=32, 
                lr=0.02, 
                workers=4, 
                agents=5,
                num=0.5, 
                kmult=0.0, 
                exp=0.7,
                opt_name="DAdSGD",
                w=None,
                kappa=0.9,
                fname=None,
                stratified=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_iterations = []
        self.test_iterations = []
        self.lr_logs = {}
        self.lambda_logs = {}
        self.loss_list = []

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.workers = workers
        self.agents = agents
        self.num = num
        self.kmult = kmult
        self.exp = exp
        self.kappa = kappa
        self.opt_name = opt_name
        self.fname = fname
        self.stratified = stratified
        self.load_data()
        self.set_opt()
        
        self.w = w
        self.criterion = torch.nn.CrossEntropyLoss()
        self.agent_setup()
        self.trainer()
        self._save()

    def _log(self, accuracy, iteration, epoch, log_interval, i):
        ''' Helper function to log accuracy values'''
        self.train_accuracy.append(accuracy)
        self.train_iterations.append(iteration + epoch * log_interval)

    def _save(self):
        with open(self.fname, mode='a') as csv_file:
            file = csv.writer(csv_file, lineterminator = '\n')
            file.writerow([f"{self.opt_name}, {self.num}, {self.kmult}, {self.batch_size}, {self.epochs}"])
            file.writerow(self.train_iterations)
            file.writerow(self.train_accuracy)
            file.writerow(self.test_iterations)
            file.writerow(self.test_accuracy)
            file.writerow(self.loss_list)
            file.writerow(["ETA"])
            for i in range(self.agents):
                file.writerow(self.lr_logs[i])
            if self.opt_name == "DLAS":
                file.writerow(["LAMBDA"])
                for i in range(self.agents):
                    file.writerow(self.lambda_logs[i])
            file.writerow([])

    def load_data(self):
        print("==> Loading Data")
        self.train_loader = {}
        self.test_loader = {}

        

        if self.dataset == 'cifar10':
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            self.class_num = 10
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            
        elif self.dataset == "mnist":
            transform_train = transforms.Compose([transforms.ToTensor(),])
            transform_test = transforms.Compose([transforms.ToTensor(),])

            self.class_num = 10
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        else:
            raise ValueError(f'{self.dataset} is not supported')

        if self.stratified:
            train_len, test_len = int(len(trainset)), int(len(testset))

            temp_train = torch.utils.data.random_split(trainset, [int(train_len//self.agents)]*self.agents)
            
            for i in range(self.agents):
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train[i], batch_size=self.batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        else:
            train_len, test_len = int(len(trainset)), int(len(testset))
            idxs = {}
            for i in range(0, 10, 2):
                arr = np.array(trainset.targets, dtype=int)
                idxs[int(i/2)] = list(np.where(arr == i)[0]) + list(np.where(arr == i+1)[0])
                shuffle(idxs[int(i/2)])
            
            percent_main = 0.5
            percent_else = (1 - percent_main) / (self.agents-1)
            main_samp_num = int(percent_main * len(idxs[0]))
            sec_samp_num = int(percent_else * len(idxs[0]))

            for i in range(self.agents):
                agent_idxs = []
                for j in range(self.agents):
                    if i == j:
                        agent_idxs.extend(sample(idxs[j], main_samp_num))
                    else:
                        agent_idxs.extend(sample(idxs[j], sec_samp_num))
                    idxs[j] = list(filter(lambda x: x not in agent_idxs, idxs[j]))
                temp_train = copy.deepcopy(trainset)
                temp_train.targets = [temp_train.targets[i] for i in agent_idxs]
                temp_train.data = [temp_train.data[i] for i in agent_idxs]
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=self.batch_size, shuffle=True)               
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    def set_opt(self):
        if self.opt_name == "DAdSGD":
            self.opt = DAdSGD
        elif self.opt_name == "DLAS":
            self.opt = DLAS
        elif self.opt_name == "CDSGD":
            self.opt = CDSGD
        elif self.opt_name == "CDSGD-P":
            self.opt = CDSGDP
        elif self.opt_name == "CDSGD-N":
            self.opt = CDSGDN
        elif self.opt_name == "DAMSGrad":
            self.opt = DAMSGrad
        elif self.opt_name == "DAdaGrad":
            self.opt = DAdaGrad

    def agent_setup(self):
        for i in range(self.agents):
            self.lr_logs[i] = []
            self.lambda_logs[i] = []

        self.agent_models = {}
        self.prev_agent_models = {}
        self.agent_optimizers = {}
        self.prev_agent_optimizers = {}

        if self.dataset == 'cifar10':
            model = CNN2()
        
        elif self.dataset == "imagenet":
            raise ValueError("ImageNet Not Supported: Low Computing Power")
        elif self.dataset == "mnist":
            model = mCNN2()

        for i in range(self.agents):
            if i == 0:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = torch.nn.DataParallel(model)
                else:
                    self.agent_models[i] = model

            else:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = copy.deepcopy(self.agent_models[0])
                else:
                    self.agent_models[i] = copy.deepcopy(model)

            self.agent_models[i].to(self.device)
            self.agent_models[i].train()

            if self.opt_name == "DAdSGD" or self.opt_name == "DLAS":
                self.prev_agent_models[i] = copy.deepcopy(model)
                self.prev_agent_models[i].to(self.device)
                self.prev_agent_models[i].train()
                self.prev_agent_optimizers[i] = self.opt(
                                params=self.prev_agent_models[i].parameters(),
                                idx=i,
                                w=self.w,
                                agents=self.agents,
                                lr=self.lr, 
                                num=self.num, 
                                kmult=self.kmult, 
                                name=self.opt_name,
                                device=self.device,
                                kappa=self.kappa,
                                stratified=self.stratified
                            )

            self.agent_optimizers[i] = self.opt(
                            params=self.agent_models[i].parameters(), 
                            idx=i,
                            w=self.w,
                            agents=self.agents,
                            lr=self.lr, 
                            num=self.num, 
                            kmult=self.kmult, 
                            name=self.opt_name,
                            device=self.device,
                            kappa=self.kappa,
                            stratified=self.stratified
                        )


    def epoch_iterations(self, epoch, dataloader):
        ''' Training function used to train model iterations for SGD
        Args:
            dataloader: dataloader as defined by pytorch for training
            epoch (int):  epoch number of training

        '''
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, grad_diff, param_diff, lambdas, all_y_vecs, old_grads = {}, {}, {}, {}, {}, {},{}
            old_y, u_tilde_5 = {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()


                if self.opt_name == "DAdSGD" or self.opt_name == "DLAS":
                    
                    self.prev_agent_optimizers[i].zero_grad()
                    prev_predicted_label = self.prev_agent_models[i](inputs)
                    prev_loss[i] = self.criterion(prev_predicted_label, labels)
                    prev_loss[i].backward()


                    if self.opt_name == "DLAS":
                        lambdas[i] = self.agent_optimizers[i].collect_lambda()

                    _, old_grads[i] = self.prev_agent_optimizers[i].collect_params()
                    grad_diff[i], param_diff[i] = self.agent_optimizers[i].compute_dif_norms(self.prev_agent_optimizers[i])

                    if torch.cuda.device_count() > 1:
                        new_mod_state_dict = OrderedDict()
                        
                        for k, v in self.agent_models[i].state_dict().items():
                            new_mod_state_dict[k[7:]] = v
                        self.prev_agent_models[i].load_state_dict(new_mod_state_dict)
                    else:
                        self.prev_agent_models[i].load_state_dict(self.agent_models[i].state_dict())

                if (self.opt_name == "DAMSGrad" or self.opt_name == "DAdaGrad") and self.running_iteration > 0:
                    u_tilde_5[i] = self.agent_optimizers[i].collect_u()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                if self.opt_name == "DAdSGD":
                    self.agent_optimizers[i].set_norms(grad_diff[i], param_diff[i])
                    self.agent_optimizers[i].step(self.running_iteration, vars=vars)
                
                elif self.opt_name == "DLAS":
                    self.agent_optimizers[i].set_norms(grad_diff[i], param_diff[i])
                    self.agent_optimizers[i].step(self.running_iteration, vars=vars, lambdas=lambdas)

                elif self.opt_name == "DAMSGrad" or self.opt_name == "DAdaGrad":
                    self.agent_optimizers[i].step(self.running_iteration, vars=vars, u_tilde_5_all=u_tilde_5)
                else:
                    self.agent_optimizers[i].step(self.running_iteration, vars=vars)
            
            if idx % log_interval == 0 and idx > 0 and epoch % 2 != 0:
                self._log(total_acc/total_count, idx, epoch, log_interval, i)
                t_acc = self.eval(self.test_loader, self.running_iteration)
                for i in range(self.agents):
                    self.lr_logs[i].append(self.agent_optimizers[i].collect_params(lr=True))
                    if self.opt_name == "DLAS":
                        self.lambda_logs[i].append(self.agent_optimizers[i].collect_lambda())

                ss = self.lr_logs[0][-1] if self.opt_name != "DLAS" else self.lambda_logs[0][-1]
                print(f"Epoch: {epoch+1}, Iteration: {self.running_iteration}, "+ 
                        f"Accuracy: {total_acc/total_count:.4f}, "+ 
                        f"Test Accuracy: {t_acc:.4f}, " + 
                        f"Loss: {tot_loss/(self.agents * log_interval):.4f}, "+
                        f"ss: {ss:.5f}, "+
                        f"Time taken: {perf_counter()-start_time:.4f}")
                        
                self.loss_list.append(tot_loss/(self.agents * log_interval))
                total_acc, total_count, tot_loss = 0, 0, 0
                self.agent_models[i].train()
                start_time = perf_counter()


        return total_acc

    def eval(self, dataloader, it=None):
        ''' Function used to evaluate training data
        Args:
            dataloader: dataloader as defined by pytorch
            it (int): iteration val
        '''
        total_acc, total_count = 0, 0

        with torch.no_grad():

            for i in range(self.agents):
                self.agent_models[i].eval()

                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    predicted_label = self.agent_models[i](inputs)

                    total_acc += (predicted_label.argmax(1) == labels).sum().item()
                    total_count += labels.size(0)

        self.test_iterations.append(it)
        self.test_accuracy.append(total_acc/total_count)

        return total_acc/total_count

    def trainer(self):
        if self.opt_name == "DAdSGD" or self.opt_name == "DLAS":
            print(f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}")
        else:
            print(f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}" +
                  f" for {self.num}, {self.kmult}")
        for i in range(self.agents):
            self.test_accuracy = []
            self.train_accuracy = []

        for i in range(self.epochs):
            accuracy = self.epoch_iterations(i, self.train_loader)

agents = 5
print(f"number of gpus: {torch.cuda.device_count()}")
w = np.array([[0.6, 0, 0, 0.4, 0],[0.2, 0.8, 0, 0, 0], [0.2, 0.1, 0.4, 0, 0.3], [0, 0, 0, 0.6, 0.4],[0, 0.1, 0.6, 0, 0.3]])

dataset = "cifar10"
epochs = 800
bs = 32

def parse_args():
    ''' Function parses command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", default=0, type=int)
    parser.add_argument("-r","--run_num", default=0, type=int)
    parser.add_argument("-s", "--stratified", action='store_true')
    return parser.parse_args()

args = parse_args()
cwd = os.getcwd()
results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

stratified = args.stratified
fname = os.path.join(results_path,f"{dataset}_e{epochs}_hom{stratified}_{args.test_num}.csv")


print(f"Test Num {args.test_num}, run num: {args.run_num}, {fname}")



if args.test_num == 0:
    DTrainer(dataset=dataset, batch_size=bs, epochs=epochs, opt_name="DAdSGD", w=w, fname=fname, stratified=stratified)
elif args.test_num == 1:
    DTrainer(dataset=dataset, batch_size=bs, epochs=epochs, opt_name="DLAS", w=w, kappa=0.37, fname=fname, stratified=stratified)
elif args.test_num == 2:
    DTrainer(dataset=dataset, batch_size=bs, epochs=epochs, opt_name="DAMSGrad", w=w, fname=fname, stratified=stratified)
elif args.test_num == 3:
    DTrainer(dataset=dataset, batch_size=bs, epochs=epochs, opt_name="DAdaGrad", w=w, fname=fname, stratified=stratified)
elif args.test_num == 4:
    DTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, opt_name="CDSGD", w=w, fname=fname, stratified=stratified)
elif args.test_num == 5:
    DTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, opt_name="CDSGD-P", w=w, fname=fname, stratified=stratified)
elif args.test_num == 6:
    DTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, opt_name="CDSGD-N", w=w, fname=fname, stratified=stratified)
