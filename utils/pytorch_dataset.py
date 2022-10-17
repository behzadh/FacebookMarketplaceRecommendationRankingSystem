import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

print("Libraries imported - ready to use PyTorch", torch.__version__)

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
        self.labels = shuffled['category_edited'].to_list()
        self.X = np.stack( shuffled['image'], axis=0 )
        self.classes = list(set(shuffled['category_edited']))
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}
        self.transform = transforms
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # is this right?
            ])

    def __getitem__(self,index):
        label = self.labels[index]
        label = self.encoder[label]
        #label = torch.as_tensor(label)

        features = self.X[index].astype(np.float32)
        features = torch.tensor(features)
        features = features.reshape(3, 64, 64)
        return (features, label)

    def __len__(self):
        return len(self.X)

    def decoding_dummies(self,index):
        decoding_labels = pd.Series(pd.Categorical(self.y_stack[self.y_stack!=0].index.get_level_values(1)))
        return decoding_labels[index]

