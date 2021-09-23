import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
from tqdm import tqdm

from MLP import MLP, weights_init
from MyDataset import MyDataset

def loadData(value_path, label_path, Batch_size, offset, context, isTrain=True):
    # load from files
    values = np.load(value_path, allow_pickle=True)     # (1) -> (1458,40)
    labels = np.load(label_path, allow_pickle=True)     # (1) -> (1458, )


    # TODO: preprocess data, scaling and standarization

    # create dataset
    dataset = MyDataset(values, labels, offset, context)

    dataloader = data.DataLoader(dataset,
                                batch_size=Batch_size,
                                shuffle=isTrain,
                                collate_fn=MyDataset.collate_fn,
                                drop_last=True)

    return values, labels, dataloader

if __name__ == "__main__":
    # data path
    # TODO: change to actual dataset name
    testdata_path = "data/toy_test_data.npy"
    testlabel_path = "data/toy_test_label.npy"      # test label may not be available
    traindata_path = "data/toy_train_data.npy"
    trainlabel_path = "data/toy_train_label.npy"
    valdata_path = "data/toy_val_data.npy"
    vallabe_path = "data/toy_val_label.npy"

    log_path = "log/"   # directory to save training checkpoint and log

    # parameters
    Epoch = 10                 # training epoch, 200
    Batch_size = 1024           # batch size, 1024
    Input_dim = 40              # input feature dimension
    Class_num = 71              # number of output class
    Context = 2                 # 5~30, need validation, extra data sampling around the interest point, make interval 2*context+1
    Offset = Context            # offset of the first batch sample index with context

    Samples_in_batch = Batch_size * (2 * Context + 1)    # actual number of samples in a batch

    Lr = 1e-4              # learning rate (for Adam, SGD need bigger)
    MILESTONES = [150]  # schedulers milestone
    MOMENTUM = 0.9      # when equals 0, no momentum
    Val_period = 10     # validate every 10 epoch

    # load data
    testdata, testlabel, testloader = loadData(testdata_path, testlabel_path, Batch_size, Offset, Context, isTrain=False)
    traindata, trainlabel, trainloader = loadData(traindata_path, trainlabel_path, Batch_size, Offset, Context, isTrain=True)   # TODO: set isTrain to True
    valdata, vallabel, valloader = loadData(valdata_path, vallabe_path, Batch_size, Offset, Context, isTrain=True)

    # check device available
    ngpu = 1  # number of gpu available
    global device
    print("Using device: " + "cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # initalize network
    mlp = MLP(input_dim=Input_dim, class_num=Class_num, context=Context).to(device)
    mlp.apply(weights_init)

    # intialize optimizer and scheduler
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=Lr, weight_decay=MOMENTUM)
    sched = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    # loss function
    criterion = nn.CrossEntropyLoss()   # not require one hot label

    # training record
    train_acc = []
    val_acc = []

    # main training loop
    for epoch in range(Epoch):
        # record train loss
        running_acc = 0.0

        # iterate batches
        for iter, data in enumerate(tqdm(trainloader)):

            values, labels = data
            values = values.to(device).float()  # send to gpu, (batch_size, 2*context+1, in)
            labels = labels.to(device).long()  # (batch_size, 1)


            if epoch == 9:
                print("check")

            values = torch.flatten(values, start_dim=1) # (batch, (2*context+1) * input_dim), flatten last 2 dimension

            optimizer.zero_grad()
            mlp.train()                 # set to train mode

            preds = mlp.forward(values)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            # compute accuracy
            preds = torch.argmax(preds, dim=1)
            running_acc += (torch.sum(preds == labels) / Batch_size).cpu().item()



        running_acc = running_acc / len(trainloader)
        print("\nEpoch: " + str(epoch + 1) + " / " + str(Epoch) + " Train loss: " + str(running_acc))
        train_acc.append(running_acc)

        # update scheduler
        sched.step()

        # validate model
        if epoch % Val_period == Val_period - 1:
            running_loss = 0.0
            for iter, data in enumerate(tqdm(valloader)):
                values, labels = data
                values = values.to(device).float()
                labels = labels.to(device).long()

                mlp.eval()          # set to validation mode
                preds = mlp.forward(values)
                loss = criterion(preds, labels)

                preds = torch.argmax(preds, dim=1)
                running_acc += (torch.sum(preds == labels) / Batch_size).cpu().item()

            running_acc = running_loss / len(valloader)
            print("\nEpoch: " + str(epoch + 1) + " / " + str(Epoch) + " Validation loss: " + str(running_loss))
            val_acc.append(running_loss)





