# this is a baseline of federated learning
import copy

import torch.optim
import time
from tqdm import tqdm
from get_data_function import get_data, average_weights
from parser import args_parser
from model import MLP, CNN
from local_update import LocalUpdate
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from test_infer import test_inference
import numpy as np
device = 'cpu'
start_time = time.time()
logger = SummaryWriter('../logs')
# load args parser
args = args_parser()

# first we need an api to get data

train_dataset, test_dataset, users_group = get_data(args)

# second we need a model
if args.model == 'cnn':
    global_model = CNN(args=args)

    # global_model = cnn(dim_in =, dim_2 =, dim_3= )

global_model.to(device)

# let model be in training mode be cautious that we did not put data on it

global_model.train()
print(global_model)

# every epochs means weight transformation
global_weights = global_model.state_dict()

# training
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 2
val_loss_pre, counter = 0, 0

for epoch in tqdm(range(args.epochs)):
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')

    global_model.train()
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    for idx in idxs_users:
        local_model = LocalUpdate(args=args, dataset=train_dataset, idx=users_group[idx], logger=logger)
        w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))

    # update global weight
    global_weights = average_weights(local_weights)
    global_model.load_state_dict(global_weights)
    
    loss_avg = sum(local_losses)/len(local_losses)
    train_loss.append(loss_avg)

    list_acc, list_loss = [], []
    global_model.eval()
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset, idx=users_group[idx], logger=logger)
        acc, loss = local_model.inference(model=global_model)
        list_acc.append(acc)
        list_loss.append(loss)

    train_accuracy.append(sum(list_acc)/len(list_acc))
    if (epoch+1) % print_every ==0:
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

# complete model and evaluate performance
test_acc, test_loss = test_inference(args, global_model, test_dataset)

print(f' \n Results after {args.epochs} global rounds of training:')
print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

import matplotlib
import matplotlib.pyplot as plt

plt.figure()
plt.title('Training Loss vs Communication rounds')
plt.plot(range(len(train_loss)), train_loss, color='r')
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds')
plt.show()

