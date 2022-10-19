import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

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

class LoadTrainTestPlot():

    '''
    A class to build a torch dataset
    '''
    def load_datasets(self, dataset):

        '''
        Loads the train and test image datasets
        '''
        # Split into training (70% and testing (30%) datasets)
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        
        # use utils.data.random_split for training/test split
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # define a loader for the training data we can iterate through in 50-image batches
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=50,
            num_workers=0,
            shuffle=True
        )
        
        # define a loader for the testing data we can iterate through in 50-image batches
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=50,
            num_workers=0,
            shuffle=True
        )
        return self.train_loader, self.test_loader

    def train(self, model, device, train_loader, optimizer, epoch, loss_criteria):
        
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

    def test(self, model, device, test_loader, loss_criteria):
        
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

    def plot_acc(self, epoch_nums, training_loss, validation_loss, dataset, model, test_loader):

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