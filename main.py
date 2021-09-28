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

def loadData(value_path, label_path, Batch_size, offset, context, isTrain=True):
    # load from files
    values = np.load(value_path, allow_pickle=True)     # (1) -> (1458,40)
    labels = np.load(label_path, allow_pickle=True)     # (1) -> (1458, )

    # create dataset
    dataset = MyDataset(values, labels, offset, context)

    dataloader = data.DataLoader(dataset,
                                batch_size=Batch_size,
                                shuffle=isTrain,
                                collate_fn=MyDataset.collate_fn,
                                pin_memory=True,
                                num_workers=8,              # up to 16
                                drop_last=isTrain)

    return values, labels, dataloader

if __name__ == "__main__":
    # data path
    traindata_path = "data/train.npy"
    trainlabel_path = "data/train_labels.npy"
    valdata_path = "data/dev.npy"
    vallabe_path = "data/dev_labels.npy"

#     traindata_path = "data/toy_train_data.npy"
#     trainlabel_path = "data/toy_train_label.npy"
#     valdata_path = "data/toy_val_data.npy"
#     vallabe_path = "data/toy_val_label.npy"

    log_path = "log/"   # directory to save training checkpoint and log

    # parameters
    Epoch = 5                 # training epoch, 50
    Batch_size = 2048           # batch size, 1024
    Input_dim = 40              # input feature dimension
    Class_num = 71              # number of output class
    Context = 10                # 5~30, need validation, extra data sampling around the interest point, make interval 2*context+1
    
    Offset = Context            # offset of the first batch sample index with context

    Samples_in_batch = Batch_size * (2 * Context + 1)    # actual number of samples in a batch

    Lr = 1e-5              # learning rate (for Adam, SGD need bigger), 1e-4
    MOMENTUM = 0.9      # when equals 0, no momentum, 0.9
    Save_period = 5     # save every 5 epoch

    # check device available
    ngpu = 1  # number of gpu available
    global device
    print("Using device: " + "cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    # load data
    print("loading data")
    traindata, trainlabel, trainloader = loadData(traindata_path, trainlabel_path, Batch_size, Offset, Context, isTrain=True)
    valdata, vallabel, valloader = loadData(valdata_path, vallabe_path, Batch_size, Offset, Context, isTrain=False)
    
    # initalize network
    mlp = MLP(input_dim=Input_dim, class_num=Class_num, context=Context).to(device)
    
    # intialize optimizer and scheduler
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=Lr, momentum=MOMENTUM)
#     optimizer = torch.optim.SGD(mlp.parameters(), lr=Lr, momentum=MOMENTUM)
    
    
    # loss function
    criterion = nn.CrossEntropyLoss()   # not require one hot label

    # training record
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    
    
    # main training loop
    for epoch in range(Epoch):
        # record train loss
        running_acc = 0.0
        trackLoss = 0.0
        print("\nEpoch: " + str(epoch + 1) + " / " + str(Epoch))
        mlp.train()                 # set to train mode
        
        # iterate batches
        for iter, data in enumerate(tqdm(trainloader)):

            values, labels = data
            values = values.to(device).float()  # send to gpu, (batch_size, 2*context+1, in)
            labels = labels.to(device).long()  # (batch_size, 1)


            preds = mlp.forward(values)
            
            optimizer.zero_grad()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()                  # update optimizer

            # compute accuracy
            preds = torch.argmax(preds, dim=1)
            running_acc += (torch.sum(1.0 * (preds == labels)) / Batch_size).cpu().item()
            trackLoss += loss.item()

        running_acc = running_acc / len(trainloader)
        trackLoss = trackLoss / len(trainloader)
        print("Train acc: " + str(running_acc * 100) + "%" + " Loss: " + str(trackLoss))
        train_acc.append(running_acc * 100)
        train_loss.append(trackLoss)

        # validate model
        mlp.eval()          # set to validation mode
        running_acc = 0.0
        trackLoss = 0.0
        print("Validating")
        for iter, data in enumerate(valloader):
            values, labels = data
            values = values.to(device).float()
            labels = labels.to(device).long()

            preds = mlp.forward(values)
            loss = criterion(preds, labels)

            preds = torch.argmax(preds, dim=1)
            running_acc += (torch.sum(1.0 * (preds == labels)) / Batch_size).cpu().item()
            trackLoss += loss.item()

        running_acc = running_acc / len(valloader)
        trackLoss = trackLoss / len(valloader)
        print("\nValidation acc: " + str(running_acc * 100) + "%" + " Loss: " + str(trackLoss))
        val_acc.append(running_acc * 100)
        val_loss.append(trackLoss)
        
        # save model
        if epoch % Save_period == Save_period - 1:
            modelpath = "log/myMLP_epoch_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': trackLoss,
            }, modelpath)


    # save loss record
    np.save("log/train_acc.npy", np.array(train_acc))
    np.save("log/train_loss.npy", np.array(train_loss))
    np.save("log/val_acc.npy", np.array(val_acc))
    np.save("log/val_loss.npy", np.array(val_loss))

  
