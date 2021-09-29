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

    log_path = "log/"   # directory to save training checkpoint and log

    # parameters
    Epoch = 20                 # training epoch, 50
    Batch_size = 2048           # batch size, 1024
    Input_dim = 40              # input feature dimension
    Class_num = 71              # number of output class
    Context = 10                # 5~30, need validation, make interval 2*context+1
    
    Offset = Context            # offset of the first batch sample index with context

    Lr = 1e-3              # learning rate (for Adam, SGD need bigger), 1e-4
    
    Factor = 0.5
    Save_period = 5     # save every 5 epoch
    Weight_decay = 1e-4   # regularization
    
    # check device available
    ngpu = 1  # number of gpu available
    global device
    print("Using device: " + "cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    # activate half precision acc
    if device == "cuda:0":
        print("activate half precision")
        scaler = torch.cuda.amp.GradScaler()
    
    
    # load data
    print("loading data")
    traindata, trainlabel, trainloader = loadData(traindata_path, trainlabel_path, Batch_size, Offset, Context, isTrain=True)
    valdata, vallabel, valloader = loadData(valdata_path, vallabe_path, Batch_size, Offset, Context, isTrain=False)
    
    # initalize network
    mlp = MLP(input_dim=Input_dim, class_num=Class_num, context=Context).to(device)
    mlp.apply(weights_init)

    # intialize optimizer and scheduler
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=Lr, weight_decay=Weight_decay)
    
    sched = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=Factor)
    
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
            values = values.to(device)  # send to gpu, (batch_size, 2*context+1, in)
            labels = labels.to(device)  # (batch_size, 1)


            preds = mlp.forward(values)
            
            optimizer.zero_grad()
            
            if device == "cuda:0":
                # activate half precision training
                with torch.cuda.amp.autocast():
                    loss = criterion(preds, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()                  # update optimizer

            # compute accuracy
            preds = torch.argmax(preds, dim=1)
            running_acc += (torch.sum(1.0 * (preds == labels)) / Batch_size).cpu().item()
            trackLoss += loss.item()

        running_acc = running_acc / len(trainloader)
        trackLoss = trackLoss / len(trainloader)
        print("Train acc: " + str(running_acc * 100) + "%" + " Loss: " + str(loss.item()))
        train_acc.append(running_acc * 100)
        train_loss.append(trackLoss)

        # validate model
        mlp.eval()          # set to validation mode
        running_acc = 0.0
        trackLoss = 0.0
        print("Validating")
        with torch.no_grad():
            for iter, data in enumerate(valloader):
                values, labels = data
                values = values.to(device)
                labels = labels.to(device)

                preds = mlp.forward(values)
                loss = criterion(preds, labels)

                preds = torch.argmax(preds, dim=1)
                running_acc += (torch.sum(1.0 * (preds == labels)) / Batch_size).cpu().item()
                trackLoss += loss.item()

            running_acc = running_acc / len(valloader)
            trackLoss = trackLoss / len(valloader)
            print("\nValidation acc: " + str(running_acc * 100) + "%" + " Loss: " + str(loss.item()))
            val_acc.append(running_acc * 100)
            val_loss.append(trackLoss)
        
            # ReduceLROnPlateau scheduler step
            sched.step(loss)
        
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
