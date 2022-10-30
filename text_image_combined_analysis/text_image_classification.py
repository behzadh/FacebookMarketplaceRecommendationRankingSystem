from text_image_dataset import ImageTextDataset, LoadTrainTestPlot
import torch
import torch.nn as nn
import pickle

class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(768, 256, kernel_size=3, stride=1, padding=1),
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
            nn.Flatten())

    def forward(self, inp):
        x = self.main(inp)
        return x

class CombinedModel(nn.Module):

    '''
    A cnn model used for text classification 
    '''

    def __init__(self, num_classes=3) -> None:

        super(CombinedModel, self).__init__()
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = resnet50.fc.out_features
        self.image_classifier = nn.Sequential(resnet50, nn.Linear(out_features, 64)).to(device)
        self.text_classifier = TextClassifier()
        self.main = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, image_features, text_features):
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

dataset = ImageTextDataset(max_length=20)
n_epochs = 8
lr = 0.001
n_classes = dataset.num_classes

load_train = LoadTrainTestPlot()

train_loader, test_loader = load_train.load_datasets(dataset)
batch_size = train_loader.batch_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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