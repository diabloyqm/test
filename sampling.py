import numpy as np


def mnist_iid(data, clients):
    per_clients = int(len(data) / clients)
    dict_clients, clients_data = {}, [i for i in range(len(data))]
    for i in range(len(clients)):
        dict_clients[i] = set(np.random.choice(clients_data, per_clients, replace=False))
        clients_data = list(set(clients_data) - dict_clients[i])
    return dict_clients


# we construct highly skewed data with 2 labels for clients each


def mnist_nonIID(dataset, num_users):
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mnist_nonIIDv2(data, clients):
    dict_users = {i: np.array([]) for i in range(clients)}
    idxs = np.arrange(len(data))
    labels = data.train_labels.numpy()
    idxs_labels = np.vstack((idxs, labels))
    list_label = list(set(labels))
    for i in range(len(set(labels))):
        random_set = set(np.random.choice((list_label), 2, replace=False))
        list_label = list(set(list_label) - random_set)
        dict_users[i] = [idxs_labels[i] for i in range(len(idxs_labels[0])) if idxs_labels[1][i] in random_set]

    return dict_users
