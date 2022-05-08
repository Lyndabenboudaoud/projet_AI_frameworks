import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class Ratings_Datset_test(Dataset):
    def __init__(self, df):
        self.df = df.reset_index()
        user_list_test = self.df.user_id.unique()
        item_list_test = self.df.recipe_id.unique()
        self.user2id_test = {w: i for i, w in enumerate(user_list_test)}
        self.item2id_test = {w: i for i, w in enumerate(item_list_test)}



    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, idx):
        user = self.user2id_test[self.df['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long)
        item = self.item2id_test[self.df['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        rating = torch.tensor(self.df['rating'][idx], dtype=torch.float)
        return user, item, rating


class NCF(nn.Module):
        
    def __init__(self, n_users, n_items, n_factors=8):
        super().__init__()
        self.user_embeddings = torch.nn.Embedding(n_users, n_factors)
        self.item_embeddings = torch.nn.Embedding(n_items, n_factors)
        self.predictor = torch.nn.Sequential(
            nn.Linear(in_features=n_factors*2, out_features=64),
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )
        
        
    def forward(self, user, item):
        

        u = self.user_embeddings(user)
        i = self.item_embeddings(item)

        # Concat the two embedding layers
        z = torch.cat([u, i], dim=-1)
        return self.predictor(z)

#testloader = DataLoader(Ratings_Datset_test(testset), batch_size=64, num_workers=2)