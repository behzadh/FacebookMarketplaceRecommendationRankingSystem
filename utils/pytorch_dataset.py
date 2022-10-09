from ast import Return
from pyexpat import features
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch

class ProductDataset(torch.utils.data.Dataset):

    '''
    A class to build a torch dataset
    '''
    def __init__(self) -> None:
        super().__init__()
        self.path = '/Users/behzad/AiCore/Facebook_Marketplace_RRS/raw_data/'
        self.test_to_train_ratio = 0.3
        self.seed = 37

        df = pd.read_pickle(self.path + 'products_w_imgs.pkl')
        shuffled = df.sample(frac=1, random_state=self.seed).reset_index()
        y_dummies = pd.get_dummiesshuffled['category_edited'])
        self.y_stack = y_dummies.stack()
        self.X = np.stack( shuffled['image'], axis=0 ) # Converts list to numpy arrays to beused for torch tensor
        self.y = y_dummies.values
        assert len(self.X) == len(self.y)

    def __getitem__(self,index):
        features = self.X[index]
        label = self.y[index]

        features = torch.tensor(features)
        features = features.reshape(64,64,3)
        return (features, label)

    def __len__(self):
        return len(self.X)

    def decoding_dummies(self,index):
        decoding_labels = pd.Series(pd.Categorical(self.y_stack[self.y_stack!=0].index.get_level_values(1)))
        return decoding_labels[index]

    def main(self):
        self.model_data()

data = ProductDataset()
#print(data[13], data.decoding_dummies(13))
dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
#print(next(iter(dataloader)))