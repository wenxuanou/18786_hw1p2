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
    traindata_path = "data/toy_train_data.npy"
    trainlabel_path = "data/toy_train_label.npy"
    valdata_path = "data/toy_val_data.npy"
    vallabe_path = "data/toy_val_label.npy"

    log_path = "log/"   # directory to save training checkpoint and log

    # parameters
    Epoch = 5                 # training epoch, 200
    Batch_size = 1024           # batch size, 1024
    Input_dim = 40              # input feature dimension
    Class_num = 71              # number of output class
    Context = 2                 # 5~30, need validation, extra data sampling around the interest point, make interval 2*context+1
    Offset = Context            # offset of the first batch sample index with context

    Samples_in_batch = Batch_size * (2 * Context + 1)    # actual number of samples in a batch

    Lr = 1e-4              # learning rate (for Adam, SGD need bigger)
    MILESTONES = [150]  # schedulers milestone
    MOMENTUM = 0.9      # when equals 0, no momentum
    Val_period = 2     # validate every 10 epoch

    # load data
    traindata, trainlabel, trainloader = loadData(traindata_path, trainlabel_path, Batch_size, Offset, Context, isTrain=True)   # TODO: set isTrain to True
    valdata, vallabel, valloader = loadData(valdata_path, vallabe_path, Batch_size, Offset, Context, isTrain=False)

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
        print("\nEpoch: " + str(epoch + 1) + " / " + str(Epoch) + " Train acc: " + str(running_acc))
        train_acc.append(running_acc)

        # update scheduler
        sched.step()


        # validate model
        if epoch % Val_period == Val_period - 1:
            # validate every 10 epoch
            running_acc = 0.0
            trackLoss = None

            for iter, data in enumerate(tqdm(valloader)):
                values, labels = data
                values = values.to(device).float()
                labels = labels.to(device).long()

                mlp.eval()          # set to validation mode
                preds = mlp.forward(values)
                loss = criterion(preds, labels)
                trackLoss = loss

                preds = torch.argmax(preds, dim=1)
                running_acc += (torch.sum(preds == labels) / Batch_size).cpu().item()

            running_acc = running_acc / len(valloader)
            print("\nEpoch: " + str(epoch + 1) + " / " + str(Epoch) + " Validation acc: " + str(running_acc))
            val_acc.append(running_acc)

            # save model
            modelpath = "log/myMLP_epoch_" + str(epoch) + "_acc_" + str(running_acc) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': trackLoss,
            }, modelpath)


    # save loss record
    np.save("log/train_acc.npy", np.array(train_acc))
    np.save("log/val_acc.npy", np.array(val_acc))

    # plot loss
    plt.figure(1)
    plt.title("train acc")
    plt.plot(train_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")

    plt.savefig("train_acc.png")

    plt.figure(2)
    plt.title("validation acc")
    plt.plot(val_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")

    plt.savefig("validation_acc.png")
