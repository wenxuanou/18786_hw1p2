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
    
    w,h = values[0].shape
    labels = np.asarray([ np.zeros((values[i].shape[0],)) for i in range(values.shape[0])])  # fake label
    
#     print("load in value shape: " + str(values.shape))
#     print("values[0] shape: " + str(values[0].shape))
#     print("load in labels shape: " + str(labels.shape))
#     print("labels[0] shape: " + str(labels[0].shape))
    
    
    # TODO: preprocess data, scaling and standarization

    # create dataset
    dataset = MyDataset(values, labels, offset, context)

    dataloader = data.DataLoader(dataset,
                                 batch_size=Batch_size,
                                 shuffle=isTrain,
                                 collate_fn=MyDataset.collate_fn,
                                 pin_memory=True,
                                 num_workers=16,         # up tp 16
                                 drop_last=False)

    return values, labels, dataloader


if __name__ == "__main__":
    # data path
    testdata_path = "data/test.npy"
    model_path = "log/myMLP_epoch_49.pt"

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
        
#         print("values shape: " + str(values.shape[1]*values.shape[2]) + ", " + str(values.shape))
#         print("input dim: " + str((2*Context+1) * Input_dim))
        assert values.shape[1]*values.shape[2] == int((2*Context+1) * Input_dim)
        
        preds = mlp.forward(values)
        preds = torch.argmax(preds, dim=1)
        preds = preds.data.cpu().numpy()
        result = np.append(result, preds, axis=0)
    
    result = np.array([np.arange(0, result.shape[0]), result])
    result = result.T
#     print(result.shape)
    np.savetxt("log/result.csv", result, delimiter=",", header="id,label", fmt="%i", comments='')