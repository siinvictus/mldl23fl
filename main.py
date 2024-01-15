import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

sys.path.append('datasets')

from entities.client import Client

from entities.centralized import Centralized
from torchvision import transforms
from entities.server import Server
from utils.args import get_parser
from torchsummary import summary

from models.cnn import CNN
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_num_classes(dataset):
    if dataset == 'femnist':
        return 62
    else:
        raise NotImplementedError


def model_init(args):
    """
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    """
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model

    if args.model == 'cnn':

        model = CNN(get_dataset_num_classes('femnist'))
        return model
    else:
        raise NotImplementedError


def get_transforms(args):
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'cnn' or args.model == 'resnet18':

        train_transforms = sstr.Compose([transforms.ToTensor(), nptr.Normalize((0.5,), (0.5,)), ])
        test_transforms = sstr.Compose([transforms.ToTensor(), nptr.Normalize((0.5,), (0.5,)), ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


def read_femnist_dir(data_dir):
    data = defaultdict(lambda: {})
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.update(cdata['user_data'])
    return data


def read_femnist_data(train_data_dir, test_data_dir=None):
    """
    If only one directory was given, the function returns the
    all_data folder content
    """
    if test_data_dir:
        return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)
    else:
        return read_femnist_dir(train_data_dir)


def get_datasets(args):

    train_transforms, test_transforms = get_transforms(args)

    train_datasets, test_datasets = [], []
    if args.federated:
        # elif args.dataset == 'femnist':
        niid = args.niid
        train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
        test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')
        train_data, test_data = read_femnist_data(train_data_dir, test_data_dir)
        total_train_data = len(train_data)
        print(f'total train data: {total_train_data}')

        for user, data in train_data.items():
            train_datasets.append(Femnist(data, train_transforms, user))
        for user, data in test_data.items():
            test_datasets.append(Femnist(data, test_transforms, user))

        return train_datasets, test_datasets,total_train_data
    else:
        all_data_dir = os.path.join('data', 'femnist', 'data', 'all_data')
        all_data = read_femnist_data(all_data_dir)
        centralized_datasets = []
        for user,data in all_data:
            centralized_datasets.append(Femnist(data, train_transforms, user))
        print(f'i m here')


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
                   'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
                   'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')}
    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
                   'test': StreamClsMetrics(num_classes, 'test')}
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_datasets, model):
    clients = [[], []]
    # define loss function criterion = nn.CrossEntropyLoss()
    idx = 0 
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model,
                                     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr),
                                     idx=idx, test_client=i == 1)
                              )
            idx += 1
    
    total_train_data = 0 
    for c in clients[0]:
        total_train_data += c.get_total_train() 
    for c in clients[0]:
        c.set_pk(total_train_data)

        
        
    print(f'Clients: Train {len(clients[0])}, Test {len(clients[1])}')
    return clients[0], clients[1]

def gen_rot_clients(args, datasets, model, angle=None):
    idx = 0
    clients = [[], []]
    if args.loo:
        print('datasets: ', len(datasets.values()), 'keys:', datasets.keys())
        for key in datasets.keys():
            if key == angle:
                for ds in datasets[key]:
                    clients[1].append(Client(args, ds, model,
                                             optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                             idx=idx, test_client=True)
                                      )
                    idx += 1
            else:
                for ds in datasets[key]:
                    clients[0].append(Client(args, ds, model,
                                             optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                             idx=idx, test_client=False)
                                      )
                    idx += 1
    else:
        indices = list(range(1000))
        sample = random.sample(indices, 700)
        split = [False if i not in sample else True for i in indices]

        for key in datasets.keys():
            for ds in datasets[key]:
                if split[idx]:
                    clients[0].append(Client(args, ds, model,
                                             optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                             idx=idx, test_client=False)
                                      )
                else:
                    clients[1].append(Client(args, ds, model,
                                             optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                             idx=idx, test_client=True)
                                      )
                idx += 1
    print(f'Clients len {len(clients)}, train {len(clients[0])}, test {len(clients[1])}')
    return clients[0], clients[1]


def fed_exec(args, model, rot_dataset=None, angle=None, train_datasets=None, test_datasets=None):

    metrics = set_metrics(args)
    # print(metrics)
    print('Gererating clients...')
    if args.rotation:
        if args.loo:
            train_clients, test_clients = gen_rot_clients(args, rot_dataset, model, angle)
        else:
            train_clients, test_clients = gen_rot_clients(args, rot_dataset, model)

    else:
        train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    print('Done.')
    print('Creating server')
    server = Server(args, train_clients, test_clients, model, metrics)
    print('Training start.....')
    server.train()


def centralized_exec(args, model):
    print('Generate datasets...')
    centralized_dataset = get_datasets(args)
    print('Done.')
    metrics = set_metrics(args)
    print('Creating centralized session')
    centralized = Centralized(data=centralized_dataset, model=model, args=args, metrics=metrics)
    print('Training start.....')
    centralized.pipeline()

def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    print('Initializing model...')
    model = model_init(args)
    model.cuda()
    if args.view_summary == True: 
        print('Summary')
        print(summary(model,(1, 28, 28)))
        #exit(1)
    print('Done.')

    if args.federated:
        print('Generate datasets...')
        if args.rotation:
            rot_dataset = get_datasets(args)
        else:
            train_datasets, test_datasets = get_datasets(args)
        print('Done.')

        if args.loo:
            angles = ['0', '15', '30', '45', '60', '75']
            for a in angles:
                print('Training Domain for angle:', a)
                fed_exec(args, model, rot_dataset=rot_dataset, angle=a)
        else:
            if args.rotatation:
                fed_exec(args, model, rot_dataset=rot_dataset)
            else:
                fed_exec(args, model, train_datasets=train_datasets, test_datasets=test_datasets)

    else:
        centralized_exec(args, model)


if __name__ == '__main__':
    path = os.getcwd()
    if 'kaggle' not in path:
        import datasets.ss_transforms as sstr
        import datasets.np_transforms as nptr
        from datasets.femnist import Femnist
    else:
        sys.path.append('datasets')
        import ss_transforms as sstr
        import np_transforms as nptr
        from femnist import Femnist

    main()
