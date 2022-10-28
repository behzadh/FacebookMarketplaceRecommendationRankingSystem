from text_image_dataset import ImageTextDataset, LoadTrainTestPlot
import torch
import torch.nn as nn
import pickle

class CombinedModel(nn.Module):

    '''
    A cnn model used for text classification 
    '''

    def __init__(self, num_classes: int, image_model, text_model) -> None:

        super(CombinedModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.main = nn.Sequential(nn.Linear(192, num_classes))

    def forward(self, image_features, text_features):
        image_features = self.image_model(image_features)
        text_features = self.text_model(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

dataset = ImageTextDataset(max_length=20)
n_epochs = 5
lr = 0.001
n_classes = dataset.num_classes

load_train = LoadTrainTestPlot()

train_loader, test_loader = load_train.load_datasets(dataset)
batch_size = train_loader.batch_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

image_model = CNN()
text_model = SecondM()
# Load state dicts
image_model.load_state_dict(torch.load('./ml_models/resnet50.pt'))
text_model.load_state_dict(torch.load('./ml_models/resnet50.pt'))

model = CombinedModel(num_classes=n_classes).to(device)
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
    torch.save(model.state_dict(), './ml_models/combined_model.pt')
    with open('./ml_models/combined_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
    load_train.plot_acc(epoch_nums, training_loss, validation_loss, dataset, model, test_loader)