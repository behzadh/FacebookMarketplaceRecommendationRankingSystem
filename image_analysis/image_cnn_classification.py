from pytorch_image_dataset import ProductDataset, LoadTrainTestPlot
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class CNN(nn.Module):

    '''
    A two layers cnn model used for image classification 
    '''

    def __init__(self, num_classes=3) -> None:

        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(12,24, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(p=0.2),
        
            nn.Flatten(),
            nn.Linear(6144, num_classes), #28800
        )

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

model = CNN(13).to(device)
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
    load_train.plot_acc(epoch_nums, training_loss, validation_loss, dataset, model, test_loader)