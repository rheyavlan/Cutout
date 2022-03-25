# Importing necessary libraries

from platform import python_version
import pdb
import argparse
import numpy as np
from tqdm import tqdm
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout

from model.resnet import ResNet18

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib
import csv
import os
import matplotlib.pyplot as plt


print("Current Numpy Version-", np.__version__)

print("Current Matplotlib Version-", matplotlib.__version__)

print("Current Python Version-", python_version())

print("Current Torch Version-", torch.__version__)

print("Current Torchvision Version-", torchvision.__version__)

# Making an out folder
if not os.path.isdir('out/'):
    os.makedirs("out/")

# Setting the model and dataset
model_options = ['resnet18']
dataset_options = ['cifar10']

# parser arguments, getting it from the user
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.dataset + '_' + args.model

print(args)

# Image Preprocessing starts

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])

if args.data_augmentation:
    print("Using basic data augmentation")
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    print("Using data augmentation CUTOUT")
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

if args.model == 'resnet18':
    print("Loading the model")
    cnn = ResNet18(num_classes=num_classes)

print("Getting the model on GPU")
cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

print("Number of Parameters: ", sum(p.numel() for p in cnn.parameters() if p.requires_grad))


scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

# filename = 'logs/' + test_id + '.csv'
print("Making csv file for logs")
filename = 'out/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        fig.savefig("batch.png");
        break
        
train_loader_dummy = train_loader
# grab a batch from both training and validation dataloader
trainBatch = next(iter(train_loader_dummy))
# visualize the training set batch
print("[INFO] visualizing training batch...")
show_batch(train_loader_dummy)    
    
def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


# Initialising these variables for storing the values that will be used for plotting
train_acc = []
testing_acc = []
train_loss = []

best_acc = 0

print("Starting Training")
for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        cnn.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)
        

    test_acc = test(test_loader)
    if test_acc > best_acc:
        #saving the model with the best test accuracy so far
        torch.save(cnn.state_dict(), 'out/' + test_id + '.pt')
        
    tqdm.write('test_acc: %.3f' % (test_acc))

#     scheduler.step(epoch)  # Use this line for PyTorch <1.4
    scheduler.step()     # Use this line for PyTorch >=1.4
    
    train_acc.append(accuracy)
    testing_acc.append(test_acc)
    train_loss.append(xentropy_loss_avg/len(train_loader))
    
    
    
    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

plt.figure(2)    
plt.plot(list(range(1, args.epochs+1)), train_acc)
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.savefig("out/train_acc.png")


plt.figure(3)   
plt.plot(list(range(1, args.epochs+1)), testing_acc)
plt.xlabel('epochs')
plt.ylabel('test accuracy')
plt.savefig("out/testing_acc.png")

plt.figure(4) 
plt.plot(list(range(1, args.epochs+1)), train_loss)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.savefig("out/train_loss.png")
    

csv_logger.close()
