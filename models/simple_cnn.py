import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

from utils.images import load_images
from models.neural_nets.cnn import Net
from utils.torchvision.utils import ReshapeTransform
from torch.utils.data.sampler import SubsetRandomSampler
from utils.torchvision.utils import EarlyStopping

import numpy as np
from sklearn.metrics import cohen_kappa_score
import tqdm

import os
import json
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

torch.manual_seed(0)
np.random.seed(0)

BATCH_SIZE = 64
IMAGE_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 20
VALIDATION_SIZE = 0.3
PATH = os.path.join(os.getcwd(), "../results/simple_cnn/")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.info(f"Session initiated on {device} device")

# Loading data
kanjis = load_images(minimum_count=10, random_seed=0, category_limit=100)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

# validation set
# obtain training indices that will be used for validation
num_train = len(x_train)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(VALIDATION_SIZE * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# transformer
transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ReshapeTransform([-1, IMAGE_SIZE, IMAGE_SIZE]),
])

logging.info(transformer)

# converting data into tensors
x_train_tensor = transformer(x_train.astype(np.uint8))
y_train_tensor = torch.tensor(y_train)

x_test_tensor = transformer(x_test.astype(np.uint8))
y_test_tensor = torch.tensor(y_test)

# setting up data loaders
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, sampler=valid_sampler)

test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# init the network
net = Net(classes=kanjis.l2i.__len__())

logging.info(net)

criterion = nn.CrossEntropyLoss()
logging.info(criterion)
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
logging.info(optimizer)

early_stopping = EarlyStopping(PATH)
logging.info(early_stopping)

logging.info(f"Running {EPOCHS} epochs with batch size of {BATCH_SIZE}...")

epoch_running_losses = []
epoch_val_losses = []
epoch_scores = []

for epoch in range(EPOCHS):  # loop over the dataset multiple times

    # train
    running_loss = 0.0
    logging.info("Running training...")
    net.train()
    for data, target in tqdm.tqdm(train_loader, unit="batches"):
        # get the inputs; data is a list of [inputs, labels]
        inp = data.reshape([BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inp)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    # validation
    logging.info("Running validation...")
    val_losses = []
    net.eval()
    for data, target in tqdm.tqdm(val_loader, unit="batches"):
        inp = data.reshape([BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE])
        # forward pass: compute predicted outputs by passing inputs to the model
        output = net(inp)
        # calculate the loss
        loss = criterion(output, target)
        # record validation loss
        val_losses.append(loss.item())

    val_loss = np.average(val_losses)
    val_losses = []

    # test
    scores = []
    logging.info("Running tests...")
    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader, unit="batches"):
            outputs = net(data.reshape([BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE]))
            _, predicted = torch.max(outputs.data, 1)
            predicted_data = predicted.data.numpy()
            target_data = target.data.numpy()
            score = cohen_kappa_score(target_data, predicted_data)
            scores.append(score)

    test_score = np.sum(scores)/scores.__len__()*100
    logging.info('Training loss after #%s/%s epoch: %.3f' % (
        str(epoch + 1), EPOCHS, running_loss))
    logging.info('Validation loss after #%s/%s epoch: %.3f' % (
        str(epoch + 1), EPOCHS, val_loss))
    logging.info('Test Accuracy after #%s/%s epoch: %.3f %%' % (
                 str(epoch+1), EPOCHS, test_score))

    epoch_running_losses.append(running_loss)
    epoch_val_losses.append(val_loss)
    epoch_scores.append(test_score)

    early_stopping(val_loss, net)

    if early_stopping.early_stop:
        logging.info("Early stopping!")
        break

# Saving data
with open(os.path.join(PATH, "results.json"), mode="w", encoding="utf8") as f:
    dump = {
        "train_loss": epoch_running_losses,
        "val_loss": epoch_val_losses,
        "kappa": epoch_scores
    }
    json.dump(dump, f)
