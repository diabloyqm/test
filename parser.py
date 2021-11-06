import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    # to be continued
    parser.add_argument('--dataset', type=str, default='mnist',help='name of dataset')

    args= parser.parse_args()
    return args
