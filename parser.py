import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # to be continued
    parser.add_argument('--dataset', type=str, default='mnist', help='name of dataset')
    parser.add_argument('--model', type=str, default='cnn', help='name of model')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd' ,help='type of optimizer')
    parser.add_argument('--class_numbers', type=int, default=10, help='number of classes')
    parser.add_argument('--chan_numbers', type=int, default=1, help='number of channels of imgs')
    parser.add_argument('--epochs', type=int, default=10, help='number of loops you want to train')
    parser.add_argument('--num_users', type=int, default=100, help='number of users')
    parser.add_argument('--local_bs', type=int, default=10, help='local batch size')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
    parser.add_argument('--iid', type=int, default=0, help='iidness')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--local_ep', type=int, default=10, help='local epoch')
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction during training')
    args = parser.parse_args()
    return args
