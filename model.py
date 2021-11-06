from torch.nn import functional

from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, precede):
        precede = precede.view(-1, precede.shape[1] * precede.shape[2] * precede.shape[-1])
        precede = self.layer_inout(precede)
        precede = self.dropout(precede)
        precede = self.relu(precede)
        precede = self.layer_hidden(precede)
        return self.softmax(precede)


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.con1 = nn.Conv2d(in_channels=args.chan_numbers, out_channels=10, kernel_size=(5, 5))
        self.con2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5))
        self.con_drop = nn.Dropout2d()
        self.l1 = nn.Linear(320, 50)
        # the reason why 320 is not clear,
        self.l2 = nn.Linear(50, args.class_numbers)
        self.relu = nn.ReLU()

    def forward(self, pre):
        pre = self.relu(functional.max_pool2d(self.con1(pre), 2))
        pre = self.relu(functional.max_pool2d(self.con_drop(self.con2(pre), 2)))
        pre = pre.view(-1, pre.shape[1]*pre.shape[2]*pre.shape[3])
        pre = self.l1(pre)
        pre = self.relu(pre)
        pre = functional.dropout(pre, training=self.training)
        pre = self.l2(pre)
        return functional.log_softmax(pre, dim=1)




