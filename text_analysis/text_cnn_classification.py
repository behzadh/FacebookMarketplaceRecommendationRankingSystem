from text_analysis.pytorch_text_dataset import TextDataset, LoadTrainTestPlot
import torch
import torch.nn as nn
import pickle

class CNN(nn.Module):

    '''
    A cnn model used for text classification 
    '''

    def __init__(self, num_classes=3) -> None:

        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

n_epochs = 5
lr = 0.001
num_classes = 13

dataset = TextDataset(max_length=20)
load_train = LoadTrainTestPlot()

train_loader, test_loader = load_train.load_datasets(dataset)
batch_size = train_loader.batch_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = CNN(num_classes=13).to(device)
loss_criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def main ():
    epoch_nums = []
    training_loss = []
    validation_loss = []

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
    torch.save(model.state_dict(), './ml_models/model_bert.pt')
    with open('./ml_models/text_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
    load_train.plot_acc(epoch_nums, training_loss, validation_loss, dataset, model, test_loader)