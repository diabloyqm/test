# this is a baseline of federated learning
import torch.optim
from tqdm import tqdm
from get_data_function import get_data
from parser import args_parser
from model import MLP, CNN
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from test_infer import test_inference
import numpy as np
device = 'cpu'

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
        local_model = 





plt.figure()
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.show()
#test
test_acc, test_loss = test_inference(args, global_model, test_dataset)
print('Test on', len(test_dataset), 'samples')
print("Test Accuracy: {:.2f}%".format(100 * test_acc))
