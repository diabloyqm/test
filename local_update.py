import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = [int(i) for i in idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        image, label = self.dataset[self.idx[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idx, logger):
        self.args = args
        self.logger = logger
        self.train_loader, self.valid_loader, self.test_loader = self.train_val_test(dataset, list(idx))
        self.device = 'cpu'
        self.loss_function = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idx):
        idx_train = idx[:int(0.8 * len(idx))]
        idx_val = idx[int(0.8 * len(idx)):int(0.9 * len(idx))]
        idx_test = idx[int(0.9 * len(idx)):]

        train_loader = DataLoader(DatasetSplit(dataset, idx_train),
                                  batch_size=self.args.local_bs, shuffle=True)
        valid_loader = DataLoader(DatasetSplit(dataset, idx_val),
                                  batch_size=int(len(idx_val)/10), shuffle=False)
        test_loader = DataLoader(DatasetSplit(dataset, idx_test),
                                 batch_size=int(len(idx_test)/10), shuffle=False)
        return train_loader, valid_loader, test_loader

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []

        # local optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr= self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.loss_function(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.train_loader.dataset), 100. * batch_idx / len(self.train_loader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.test_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            batch_loss = self.loss_function(outputs, labels)
            loss += batch_loss.item()

            _, prediction = torch.max(outputs, 1)
            prediction = prediction.view(-1)
            correct += torch.sum(torch.eq(prediction, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss








