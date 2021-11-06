# this is the frame work of federated learning
from get_data_function import get_data

device = 'cpu'
# load args parser
from parser import args_parser

# first we need an api to get data
args = args_parser()
train_dataset, test_dataset = get_data()

# second we need a model

# global_model = your_model(dim_in =, dim_2 =, dim_3= )

# global_model.to(device)

# let model be in training mode be cautious that we did not put data on it
# global_model.train()

# train the model e.g. sgd method
# optimizer = torch.optim.....

# an important api that standardize dataset (into tensor)
# DataLoader(train_dataset, batch_size=, shuffle=)

#  tools for evaluation
# torch.nn.NLLLoss().to(device)

# main epochs
