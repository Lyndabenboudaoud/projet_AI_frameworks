from model import Ratings_Datset_test
from model import NCF
import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recepies sentiment')
    
    parser.add_argument("-m", "--model", required = True, help="ML Model", default="A random string")

    parser.add_argument("-d", "--data", required = True, help="data file", default="A random string")

    args = parser.parse_args()

    model = torch.load(args.model)

    testset = pd.read_csv(args.data, delimiter = ',')

    testloader = DataLoader(Ratings_Datset_test(testset), batch_size=10, num_workers=2)

    users, recipes, r = next(iter(testloader))

    y = model(users, recipes)*5

    print("ratings", r[:10].data)
    print("predictions:", y.flatten()[:10].data)
