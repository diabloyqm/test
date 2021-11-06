# this is the frame work of federated learning
from get_data_function import get_data
from parser import args_parser
from model import MLP, CNN

device = 'cpu'

# load args parser
args = args_parser()

# first we need an api to get data

train_dataset, test_dataset = get_data(args)

# second we need a model
if args.model == 'cnn':
    global_model = cnn(args=args)

    # global_model = cnn(dim_in =, dim_2 =, dim_3= )

global_model.to(device)
global_model.train()
print(global_model)

# let model be in training mode be cautious that we did not put data on it
# global_model.train()

# train the model e.g. sgd method
# optimizer = torch.optim.....

# an important api that standardize dataset (into tensor)
# DataLoader(train_dataset, batch_size=, shuffle=)

#  tools for evaluation
# torch.nn.NLLLoss().to(device)

# main epochs
