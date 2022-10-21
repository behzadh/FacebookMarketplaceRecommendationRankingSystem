import os
from pytorch_dataset import ProductDataset, LoadTrainTestPlot
import torch
import torch.nn as nn
import pickle

class ResNet50(nn.Module):

    '''
    ResNet50 class is a pre-trained model tuned for our image classification  
    '''

    def __init__(self, num_classes=3) -> None:

        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features #It'd be 2048 for resnet50
        self.linear = nn.Linear(out_features, num_classes).to(device)
        self.layers = nn.Sequential(self.resnet50, self.linear).to(device)

    def forward(self, x):
        return self.layers(x)

batch_size = 4
n_epochs = 3
lr = 0.001
num_classes = 13

dataset = ProductDataset()
load_train = LoadTrainTestPlot()

train_loader, test_loader = load_train.load_datasets(dataset)
batch_size = train_loader.batch_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = ResNet50(13).to(device)
loss_criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def main ():
    epoch_nums = []
    training_loss = []
    validation_loss = []
    if not os.path.exists('/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/weights'): os.makedirs('/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/weights')
    if not os.path.exists(f'/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/weights/{type(model).__name__}'): os.makedirs(f'/Users/behzad/AiCore/Facebook_Marketplace_RRS/ml_models/weights/{type(model).__name__}')

    for epoch in range(n_epochs):
        train_loss = load_train.train(model, device, train_loader, optimizer, epoch, loss_criteria)
        test_loss = load_train.test(model, device, test_loader, loss_criteria)
        epoch_nums.append(epoch + 1)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
       # torch.save(model.state_dict(), f'./ml_models/weights/{type(model).__name__}/resnet50_{epoch+1}.pt')
    return epoch_nums, training_loss, validation_loss

if __name__ == "__main__":
    epoch_nums, training_loss, validation_loss = main()
    torch.save(model.state_dict(), './ml_models/resnet50.pt')
    with open('./ml_models/image_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
    load_train.plot_acc(epoch_nums, training_loss, validation_loss, dataset, model, test_loader)