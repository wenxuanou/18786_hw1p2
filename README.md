# 18786_hw1p2

## Packeages used

pytorch: For dataset and model training.

numpy: For computation.

tqdm: For progress bar.

matplotlib: For plotting training accuracy.

## How to use

Before running, need to create /log and /data directory in the root directory for training log and data storage.

Run model training with command: python main.py (need to configure data path)

Run model testing with command: python testModel.py (need to configure data path)

Plot training accuracy: python plotacc.py (need to configure data path)

## File desciption

main.py: main program for model training.

testModel.py: load a trained model and produce test result. Parameters need to match those in main.py.

MLP.py: my MLP model.

MyDataset.py: function to load and construct dataset from file.


## Training parameter used

Epoch = 30                  # training epoch
Batch_size = 2048           # batch size
Input_dim = 40              # input feature dimension
Class_num = 71              # number of output class
Context = 20                # data sample context
Offset = Context            # offset of the first batch sample index with context
Lr = 1e-3                   # learning rate
Factor = 0.1                # ReduceLROnPlateau scheduler decay factor
Save_period = 5             # save model every 5 epoch
Weight_decay = 1e-6         # Adam regularization

Best accuracy: 0.7895


