from torchvision import datasets, transforms
import numpy as np
from sampling import mnist_iid, mnist_nonIID


def get_data(args):
    if args.dataset == 'mnist':
        mnist_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.MNIST('../data/mnist/', train=True, download=True, transform=mnist_trans)
        test_data = datasets.MNIST('../data/mnist/', train=False, download=True, transform=mnist_trans)
        if args.iid == 1:
            users_group = mnist_iid(train_data, args.num_users)
        elif args.iid == 0:
            users_group = mnist_nonIID(train_data, args.num_users)
    elif args.dataset == 'cifar':
        cifar_trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR10('../data/cifar/', train=True, download=True, transform=cifar_trans)
        test_data = datasets.CIFAR10('../data/cifar/', train=False, download=True, transform=cifar_trans)
        if args.iid == 1:
            users_group = mnist_iid(train_data, args.num_users)
        elif args.iid == 0:
            users_group = mnist_nonIID(train_data, args.num_users)
    return train_data, test_data, users_group

# 超参数
# clients = 10
#
# # this is an iid sampling method
# per_clients = int(len(mnist_data) / clients)
# clients_dict, index_clients_dict = {}, [i for i in range(len(mnist_data))]
# for i in range(clients):
#     clients_dict[i] = set(np.random.choice(index_clients_dict, per_clients, replace=True))
#     index_clients_dict = list(set(index_clients_dict) - clients_dict[i])
# print(clients)
