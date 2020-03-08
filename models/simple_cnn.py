import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

from utils.images import load_images
from models.neural_nets.cnn import Net

import numpy as np

# hyper params.

BATCH_SIZE = 64
IMAGE_SIZE = 64

# Loading data
kanjis = load_images(minimum_count=5, random_seed=0, category_limit=10)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

# converting data into tensors
x_train_tensor = torch.tensor(x_train[:x_train.shape[0]//BATCH_SIZE*BATCH_SIZE], dtype=torch.float32)
y_train_tensor = torch.tensor(y_train[:x_train.shape[0]//BATCH_SIZE*BATCH_SIZE])
x_test_tensor = torch.tensor(x_test[:x_test.shape[0]//BATCH_SIZE*BATCH_SIZE], dtype=torch.float32)
y_test_tensor = torch.tensor(y_test[:x_test.shape[0]//BATCH_SIZE*BATCH_SIZE])

# setting up data loaders
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# init the network
net = Net(classes=kanjis.l2i.__len__())

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs: torch.Tensor

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.reshape([BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE]))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images.reshape([BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE]))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy after #%s epoch: %.3f %%' % (
        str(epoch+1), 100 * correct / total))
