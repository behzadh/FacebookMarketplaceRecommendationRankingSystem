import pandas as pd
import torch
import torchvision.transforms as transforms
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
        self.X = shuffled['id']
        self.classes = list(set(shuffled['category_edited']))
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}
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

        imgs = Image.open('/Users/behzad/AiCore/fb_raw_data/cleaned_images/' + self.X[index] + '_resized.jpg')
        imgs = self.transform(imgs)
        return (imgs, label)

    def __len__(self):
        return len(self.X)


