

from pytorch_dataset import ProductDataset, LoadTrainTestPlot
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

class ResNet50(nn.Module):

    '''
    A 2 layer cnn model used for image classification 
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
n_epochs = 2
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

    #writer = SummaryWriter()

    for epoch in range(n_epochs):
        train_loss = load_train.train(model, device, train_loader, optimizer, epoch, loss_criteria)
        test_loss = load_train.test(model, device, test_loader, loss_criteria)
        epoch_nums.append(epoch + 1)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
    return epoch_nums, training_loss, validation_loss

if __name__ == "__main__":
    epoch_nums, training_loss, validation_loss = main()
    torch.save(model.state_dict(), './raw_data/resnet50.pt')
    with open('./raw_data/image_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
    load_train.plot_acc(epoch_nums, training_loss, validation_loss, dataset, model, test_loader)