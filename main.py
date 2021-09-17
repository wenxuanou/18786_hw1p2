import torch
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


import numpy as np

from MLP import MLP
from MyDataset import MyDataset

def loadData(value_path, label_path, batch_size, isTrain=True):
    # load from files
    values = np.load(value_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)

    # create dataset
    dataset = MyDataset(values, labels)

    dataloader = data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=isTrain,
                                       collate_fn=MyDataset.collate_fn)

    return values, labels, dataloader

if __name__ == "__main__":
    # data path
    # TODO: change to actual dataset name
    testdata_path = "data/toy_test_data.npy"
    testlabel_path = "data/toy_test_label.npy"
    traindata_path = "data/toy_train_data.npy"
    trainlabel_path = "data/toy_train_label.npy"
    valdata_path = "data/toy_val_data.npy"
    vallabe_path = "data/toy_val_label.npy"

    # parameters
    Batch_size = 2

    # load data
    testdata, testlabel, testloader = loadData(testdata_path, testlabel_path, Batch_size, isTrain=False)
    traindata, trainlabel, trainloader = loadData(traindata_path, trainlabel_path, Batch_size, isTrain=True)
    valdata, vallabel, valloader = loadData(valdata_path, vallabe_path, Batch_size, isTrain=True)

    for i, batch in enumerate(trainloader):
        print("Batch", i, ":\n", batch, "\n")