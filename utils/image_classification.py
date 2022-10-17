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
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)
#print(next(iter(dataloader)))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ImageClassifier(torch.nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = torch.nn.Linear(out_features, num_classes).to(device)
        self.layers = torch.nn.Sequential(self.resnet50, self.linear).to(device)

    def forward(self, x):
        return self.layers(x)

model = ImageClassifier()
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