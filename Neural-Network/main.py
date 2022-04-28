import argparse
import os

import numpy as np
import torch

from train import DTrainer

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