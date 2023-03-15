# Necessary libraries
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json
# defining Mandatory and Optional Arguments 
parser = argparse.ArgumentParser (description = "Parser of training script")

parser.add_argument('data_dir', type=str, default='flowers', help='data root')
parser.add_argument('--save_dir', type=str, default='checkpoint', help='save the trained model to a checkpoint')
parser.add_argument('--arch', type=str, default='vgg13', help='CNN architecture')
parser.add_argument('--lrn', type=float, default=0.001, help='learning_rate')
parser.add_argument('--hidden_unit1', type=int, default=1024, help='hidden_unit1')
parser.add_argument('--hidden_unit4', type=int, default=102, help='hidden_unit4')
parser.add_argument('--epochs', type=int, default=6, help='num of epochs')
parser.add_argument('--GPU', type=str, default='GPU', help='use GPU for training')

#setting values data loading
args = parser.parse_args ()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#defining device: either cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
  # Define your transforms for the training, validation, and testing sets
if data_dir:
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test':transforms.Compose ([
             transforms.Resize((224, 224)),
                    transforms.ToTensor (),
                    transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                    ])}

#  Load the datasets with ImageFolder 
train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
# Using the image datasets and the trainforms, define the dataloaders

# training data
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
# validation
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
# test
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
#mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    #    /////////////////////////////// check error in loop

# def load_model (arch, hidden_units):
#     if arch == 'vgg16': #setting model based o  
#         model = models.vgg16 (pretrained = True)       
def network(arch= args.arch ,lrn = args.lrn ,  hidden_unit1 =  args.hidden_unit1 , hidden_unit4 = args.hidden_unit4):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    
    else:
        print("Please try for vgg13 or densenet121")

    for param in model.parameters():
        param.requires_grad = False

#     from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(input_size, hidden_unit1)),
                ('relu1', nn.ReLU()),
                ('fc1', nn.Linear(hidden_unit1, 316)),
                ('relu2',nn.ReLU()),
                ('fc2',nn.Linear(316, 158)),
                ('relu3',nn.ReLU()),
                ('fc3',nn.Linear(158, hidden_unit4)),
#                 ('fc3',nn.Linear(158, 102)),
                ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lrn)
    model.cuda()
# ////////////////////////////////////////////////// used return model, arch
    return model, optimizer,criterion   
# model,optimizer,criterion = network(choice = 'vgg16')
model,optimizer,criterion = network(arch = 'vgg13')
criterion = nn.NLLLoss()
model.to (device) #device can be either cuda or cpu
#setting number of epochs to be run
if args.epochs:
    epochs = args.epochs
else:  
    epochs =6
    
print_every = 40
steps = 0
model.to('cuda')
for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
            
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
            
        if steps % print_every == 0:
            
            model.eval()
            v_lost = 0
            v_accuracy=0
                
            for ii, (inputs2,labels2) in enumerate(validloader):
                optimizer.zero_grad()
                inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                model.to('cuda')
                    
                with torch.no_grad():
                    outputs = model.forward(inputs2)
                    v_lost = criterion(outputs,labels2)
                    ps = torch.exp(outputs).data
                    equality = (labels2.data == ps.max(1)[1])
                    v_accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
            v_lost = v_lost / len(validloader)
            v_accuracy = v_accuracy /len(validloader)
                
            print("Epoch: {}/{}... ".format(e+1, epochs),
                 "Training Loss: {:.4f}".format(running_loss/print_every),
                 "Validation Loss: {:.4f}".format(v_lost),
                 "Validation Accuracy: {:.4f}".format(v_accuracy * 100))

            running_loss = 0


#saving trained Model
model.to ('cpu') #no need to use cuda for saving/loading model.
# Save the checkpoint
model.class_to_idx = train_datasets.class_to_idx #saving mapping between predicted class and class name,
#second variable is a class name in numeric

#creating dictionary for model saving
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': args.arch,
              'mapping':    model.class_to_idx
             }
#saving trained model for future use
if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')
