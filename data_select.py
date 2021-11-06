from torchvision import datasets, transforms
import numpy as np
mnist_trans = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST('../data/mnist', train= True, download=True, transform=mnist_trans)
mnist_data
# 超参数
clients = 10

# this is an iid sampling method
per_clients = int(len(mnist_data) / clients)
clients_dict, index_clients_dict = {}, [i for i in range(len(mnist_data))]
for i in range(clients):
    clients_dict[i] = set(np.random.choice(index_clients_dict, per_clients, replace=True))
    index_clients_dict = list(set(index_clients_dict) - clients_dict[i])
print(clients)

