import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from MLP import MLP, weights_init
from MyDataset import MyDataset

def loadTestData(value_path, Batch_size, offset, context, isTrain=True):
    # load from files
    values = np.load(value_path, allow_pickle=True)  # (1) -> (1458,40)
    labels = values  # (1) -> (1458, )

    # TODO: preprocess data, scaling and standarization

    # create dataset
    dataset = MyDataset(values, labels, offset, context)

    dataloader = data.DataLoader(dataset,
                                 batch_size=Batch_size,
                                 shuffle=isTrain,
                                 collate_fn=MyDataset.collate_fn,
                                 pin_memory=True,
                                 num_workers=8,
                                 drop_last=True)

    return values, labels, dataloader


if __name__ == "__main__":
    # data path
    testdata_path = "data/test.npy"
    model_path = "log/myMLP_epoch_3_acc_0.16421473978136883.pt"

    # parameters
#     Epoch = 5  # training epoch, 200
    Batch_size = 1024  # batch size, 1024    # need to match main.py
    Input_dim = 40  # input feature dimension, 40
    Class_num = 71  # number of output class, 71
    Context = 15  # 5~30, extra data sampling around the interest point, make interval 2*context+1   #need to match main.py
    Offset = Context  # offset of the first batch sample index with context

    # load test data
    testdata, _, testloader = loadTestData(testdata_path, Batch_size, Offset, Context, isTrain=False)   # drop labels

    # check device available
    ngpu = 1  # number of gpu available
    global device
    print("Using device: " + "cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # initialize mlp
    mlp = MLP(input_dim=Input_dim, class_num=Class_num, context=Context).to(device)

    # load model
    checkpoint = torch.load(model_path)
    mlp.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # test model
    mlp.eval()
    result = np.array([])
    for data in tqdm(testloader):
        values,_ = data
        values = values.to(device).float()
        preds = mlp.forward(values)
        preds = torch.argmax(preds, dim=1)
        preds = preds.data.cpu().numpy()
        result = np.append(result, preds, axis=0)

    np.savetxt("log/result.csv", result, delimiter=",")