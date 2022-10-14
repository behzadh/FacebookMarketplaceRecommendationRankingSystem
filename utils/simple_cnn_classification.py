
import numpy as np
from pytorch_dataset import ProductDataset
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


data = ProductDataset()
#print(data[13], data.decoding_dummies(13))
batch_size = 13
n_epochs = 1
lr = 0.001
num_classes = 13
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
#print(next(iter(dataloader)))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CNN(torch.nn.Module):

    '''
    
    '''
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
        
            torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            
            torch.nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            #torch.nn.Conv2d(3, 8, 5),
            torch.nn.Flatten(),
            torch.nn.Linear(16384, 1024), #28800
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,13)
        )

    def forward(self, x):
        return self.layers(x)

model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

writer = SummaryWriter()

losses = []
batch_inx = 0

for epoch in range(n_epochs):
    hist_accuracy = []
    accuracy = 0
    # pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    # for i, (data, labels) in pbar:
    for batch in dataloader:
        data, labels = batch
        #print(len(labels))
        if len(labels) != 13: continue
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        accuracy = torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels)
        hist_accuracy.append(accuracy)
        losses.append(loss.item())
        writer.add_scalar('loss', loss.item(), batch_inx)
        writer.add_scalar('accuracy', accuracy, batch_inx)
        batch_inx += 1
        # pbar.set_description(f"Epoch = {epoch+1}/{n_epochs}. Acc = {round(torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels), 2)}, Total_acc = {round(np.mean(hist_accuracy), 2)}, Losses = {round(loss.item(), 2)}" )
        # print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")
        # print('-'*20)
        # print(f"Accuracy: {torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels)}")
        # print('-'*20)