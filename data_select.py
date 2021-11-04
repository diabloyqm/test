from torchvision import datasets, transforms
import numpy as np
mnist_trans = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST('../data/mnist',transform=mnist_trans)

clients = 10
per_clients = int(len(mnist_data) / clients)
clients_dict, index_clients_dict = {}, [i for i in range(len(mnist_data))]
for i in range(clients):
    clients_dict[i] = set(np.random.choice(index_clients_dict, per_clients, replace=False))
    index_clients_dict = list(set(index_clients_dict) - clients_dict[i])


