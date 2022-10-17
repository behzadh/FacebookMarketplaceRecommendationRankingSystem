
import numpy as np
import pandas as pd
from pytorch_dataset import ProductDataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix

batch_size = 4
n_epochs = 2
lr = 0.001
num_classes = 13

dataset = ProductDataset()

def load_dataset():

    '''
    Loads the train and test image datasets
    '''
    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    
    # use utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=True
    )
    
    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=True
    )
    return train_loader, test_loader

train_loader, test_loader = load_dataset()
batch_size = train_loader.batch_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CNN(nn.Module):

    '''
    A 2 layer cnn model used for image classification 
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

model = CNN(13).to(device)
loss_criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model, device, train_loader, optimizer, epoch):
    
    model.train() # Set the model to training mode
    train_loss = 0
    print("Epoch:", epoch + 1)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_criteria(output, target)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        #print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def test(model, device, test_loader):
    
    model.eval() # Switch the model to evaluation mode (so we don't backpropagate or drop)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += loss_criteria(output, target).item()          
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return avg_loss

def plot_acc(epoch_nums, training_loss, validation_loss):

    plt.figure(figsize=(7,7))
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

    # Defining Labels and Predictions
    truelabels = []
    predictions = []
    model.eval()
    print("Getting predictions from test set...")
    for data, target in test_loader:
        for label in target.data.numpy():
            truelabels.append(label)
        for prediction in model(data).data.numpy().argmax(1):
            predictions.append(prediction) 

    # Plot the confusion matrix
    cm = confusion_matrix(truelabels, predictions)
    tick_marks = np.arange(len(dataset.classes))

    df_cm = pd.DataFrame(cm, index = dataset.classes, columns = dataset.classes)
    plt.figure(figsize = (7,7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel("Predicted Shape", fontsize = 20)
    plt.ylabel("True Shape", fontsize = 20)
    plt.show()

def main ():
    epoch_nums = []
    training_loss = []
    validation_loss = []

    #writer = SummaryWriter()

    for epoch in range(n_epochs):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
    return epoch_nums, training_loss, validation_loss

if __name__ == "__main__":
    epoch_nums, training_loss, validation_loss = main()
    plot_acc(epoch_nums, training_loss, validation_loss)