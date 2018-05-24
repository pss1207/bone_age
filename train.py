import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from model import define_network
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.metrics import mean_absolute_error
from data_loader import CustomTrainImageFolder, CustomTestImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import time
import copy

dataset_path = '/media/hdd/data/rsna-bone-age'
IMG_SIZE = 512
BATCH_SIZE = 8

train_csv = pd.read_csv(dataset_path + '/boneage-training-dataset.csv')
test_csv = pd.read_csv(dataset_path + '/boneage-test-dataset.csv')

train_data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomRotation((-5, 5)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

test_data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

train_path = dataset_path + '/boneage-training-dataset'
test_path = dataset_path + '/boneage-test-dataset'

train_dataset = CustomTrainImageFolder(train_path, train_csv, train_data_transforms)
test_dataset = CustomTestImageFolder(test_path, test_csv, test_data_transforms)

val_size = 0.15
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(val_size * num_train))
train_idx, val_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                                           sampler=train_sampler,
                                           num_workers=4)
val_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                                           sampler=val_sampler,
                                         num_workers=4)


test_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                                          num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = 999999.9

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Training Phase

        # Each epoch has a training and validation phase
        scheduler.step()
        model.train()  # Set model to training mode

        running_loss = []
        running_mae = []
        # Iterate over data.
        step = 0
        for inputs, male, boneage in train_loader:
            step += 1
            inputs = inputs.to(device)
            male = torch.from_numpy(np.array(male)).type(torch.FloatTensor)
            male = torch.unsqueeze(male, 1)
            male = male.to(device)
            boneage = torch.from_numpy(np.array(boneage)).type(torch.FloatTensor)
            boneage = torch.unsqueeze(boneage, 1)
            boneage = boneage.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                output = model([inputs, male])
                loss = criterion(output, boneage)
                mae = mae_months(boneage.detach(), output.detach())

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss.append(loss.item())
            running_mae.append(mae)

            #if step%32==0:
                #print('Epoch: {} {}/{} Loss: {:.4f} MAE_Months: {:.4f}'.format(epoch, step*inputs.size(0), num_train, np.mean(running_loss), np.mean(running_mae)))

        epoch_loss = np.mean(running_loss)
        epoch_mae = np.mean(running_mae)

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_mae))


        # Validation Phase

        # Each epoch has a training and validation phase
        model.eval()  # Set model to training mode

        running_loss = []
        running_mae = []
        # Iterate over data.
        step = 0
        for inputs, male, boneage in val_loader:
            step += 1
            inputs = inputs.to(device)
            male = torch.from_numpy(np.array(male)).type(torch.FloatTensor)
            male = torch.unsqueeze(male, 1)
            male = male.to(device)
            boneage = torch.from_numpy(np.array(boneage)).type(torch.FloatTensor)
            boneage = torch.unsqueeze(boneage, 1)
            boneage = boneage.to(device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                output = model([inputs, male])
                loss = criterion(output, boneage)
                mae = mae_months(boneage, output)

            # statistics
            running_loss.append(loss.item())
            running_mae.append(mae)

        epoch_loss = np.mean(running_loss)
        epoch_mae = np.mean(running_mae)

        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_mae))

        # deep copy the model
        if epoch_mae < best_mae:
            best_mae = epoch_mae
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_model.pth')
            print ('Best Epoch: {}'.format(epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best MAE Months: {:4f}'.format(best_mae))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = define_network()
for i, param in model_ft.named_parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 1)
'''
model_ft.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.Tanh(),
                nn.Dropout(0.25),
                nn.Linear(1024, 1)
)
'''
# Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
ct = []
for name, child in model_ft.named_children():
    if "Mixed_5b" in ct:
        for params in child.parameters():
            params.requires_grad = True
    ct.append(name)

for child in model_ft.children():
  for name_2, params in child.named_parameters():
    print(name_2, params.requires_grad)

model_ft = model_ft.to(device)

criterion = nn.MSELoss()

def mae_months(target, pred):
    return mean_absolute_error(target, pred)

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001)

# Decay LR by a factor of 0.1 every epoch
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=500)