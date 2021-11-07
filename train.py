# this is the frame work of federated learning
import torch.optim
from tqdm import tqdm
from get_data_function import get_data
from parser import args_parser
from model import MLP, CNN
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from test_infer import test_inference
device = 'cpu'

# load args parser
args = args_parser()

# first we need an api to get data

train_dataset, test_dataset = get_data(args)

# second we need a model
if args.model == 'cnn':
    global_model = CNN(args=args)

    # global_model = cnn(dim_in =, dim_2 =, dim_3= )

global_model.to(device)

# let model be in training mode be cautious that we did not put data on it

global_model.train()
print(global_model)

# train the model e.g. sgd method
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=0.5)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-4)

# an important api that standardize dataset (into tensor)
# DataLoader(train_dataset, batch_size=, shuffle=)
data_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

#  tools for evaluation
# torch.nn.NLLLoss().to(device)
loss_function = torch.nn.NLLLoss().to(device)

# main epochs
epoch_loss = []
for epoch in tqdm(range(args.epochs)):
    batch_loss = []

    for batch_index, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = loss_function(global_model(images), labels)
        loss.backward()
        optimizer.step()

        if batch_index % 50==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch+1, batch_index * len(images), len(data_loader.dataset),100. * batch_index / len(data_loader), loss.item()))
        batch_loss.append(loss.item())

    loss_avg = sum(batch_loss)/len(batch_loss)
    print('\nTrain_loss:', loss_avg)

plt.figure()
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.show()

#test
test_acc, test_loss = test_inference(args, global_model, test_dataset)
print('Test on', len(test_dataset), 'samples')
print("Test Accuracy: {:.2f}%".format(100 * test_acc))


