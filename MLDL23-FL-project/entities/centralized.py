import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import time

import os
import sys
path = os.getcwd()
if 'kaggle' not in path:
    from datasets.femnist import Femnist
else:
    sys.path.append('datasets')
    from femnist import Femnist



IMAGE_SIZE = 28






class Centralized:

    def __init__(self, data_path, model, optimizer, criterion, device, transforms):
        self.path = data_path
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.transforms = transforms

    def n_classes(self, batch):
        return batch['class'].unique().shape[0]
    def data_parser(self, df):
        """
        takes a dataframe sorted by writers and unpacks the data
        return: dataframe with two columns [img,class]
        """
        transpose = df.T
        out_dict = dict()
        idx = 0
        for _, row in transpose.iterrows():
            row_x = row.x
            for i, val in enumerate(row.y):
                out_dict[idx] = [np.array(row_x[i], dtype=np.float16), val]
                idx += 1
        out_df = pd.DataFrame(out_dict).T
        out_df = out_df.rename(columns={0: 'img', 1: 'class'})
        return out_df

    def get_data(self):
        df = pd.DataFrame()
        print('loading files.....')
        print(f'file path {self.path}')
        start_t = time.time()
        for dirname, _, filenames in os.walk(self.path):
            print(f'dir name {dirname}  and filenames{filenames}')
            for filename in filenames:
                print(filename)
                data = json.load(open(os.path.join(dirname, filename)))

                temp_df = pd.DataFrame(data['user_data'])
                temp_df = temp_df.reset_index(drop=True)
                df = pd.concat([df, temp_df], axis=1)  # ignore_index=True
        end_t = time.time()
        print(f'The end time of the data reading was {end_t - start_t}.')
        df = df.rename(index={0: "x", 1: "y"})
        return df

    def train_test_tensors(self, batch):
        convert_tensor = transforms.ToTensor()
        X_train, X_val, y_train, y_val = train_test_split(batch['img'], batch['class'], test_size=0.2, random_state=42)
        torch_train = Femnist(
            {'x': X_train.tolist(), 'y': y_train.tolist()},
            self.transforms, '')
        torch_test = Femnist(
            {'x': X_val.tolist(), 'y': y_val.tolist()},
            self.transforms, '')

        return torch_train, torch_test

    def training(self, torch_train):

        train_loader = DataLoader(torch_train, batch_size=64, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for epoch in range(5):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print('Finished Training')

    def accuracy_of_model(self, val_loader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    """
    def accuracy_for_class(self, val_loader, classes):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in range(classes)}
        total_pred = {classname: 0 for classname in range(classes)}

        # again no gradients needed
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    """
    def pipeline(self):
        print('loading data')
        out_df = self.get_data()
        print('preprocessing')
        # dataframe of the dataset
        df = self.data_parser(out_df)
        del out_df
        print('Done')
        #n_classes = self.n_classes(df)
        # train and test tensors
        torch_train, torch_test = self.train_test_tensors(df)
        print('Training')
        self.training(torch_train)
        print('Done.')
        # printing accuracy
        val_loader = DataLoader(torch_test, batch_size=64, shuffle=False)
        print('Validating')
        self.accuracy_of_model(val_loader)

