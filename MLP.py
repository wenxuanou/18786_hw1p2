import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=40, class_num=71, context=1):
        super(MLP, self).__init__()

        # take in (batch, (2*context+1), input_dim)
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),        # flatten last 2 dimension, (batch, (2*context+1), input_dim)
            nn.Linear(int((2*context+1) * input_dim), int((2*context+1) * input_dim)),
            nn.BatchNorm1d(int((2*context+1) * input_dim)),
            nn.ReLU(True),
            nn.Linear(int((2*context+1) * input_dim), int((2*context+1) * input_dim)),
            nn.BatchNorm1d(int((2*context+1) * input_dim)),
            nn.ReLU(True),
            nn.Linear(int((2 * context + 1) * input_dim), int((2 * context + 1) * input_dim)),
            nn.BatchNorm1d(int((2 * context + 1) * input_dim)),
            nn.ReLU(True),
            
#             nn.Linear(int((2*context+1) * input_dim), int((2*context+1) * input_dim / 20)),   # ((2*context+1) * input_dim, 1024)
#             nn.BatchNorm1d(int((2*context+1) * input_dim / 20)),
#             nn.ReLU(True),
#             nn.Linear(int((2*context+1) * input_dim / 20), int((2*context+1) * input_dim / 200)),          # (1024, 512)
#             nn.BatchNorm1d(int((2*context+1) * input_dim / 200)),
#             nn.ReLU(True),
#             nn.Linear(int((2*context+1) * input_dim / 200), 256),          # (512, 256)
#             nn.BatchNorm1d(256),
#             nn.ReLU(True),
#             nn.Linear(256, 128),          # (256, 128)
#             nn.BatchNorm1d(128),
#             nn.ReLU(True),
            
            nn.Linear(int((2*context+1) * input_dim), class_num)    # (128,71)
            
            
#             nn.Linear(128, class_num)    # (128,71)
        )

    def forward(self, x):
        y = self.mlp(x)     # y: (batch, 71)
        return y

# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # unitform distribution
        # value sampled from u~(-bound, bound), where bound = 3 * gain / sqrt(fan_mode)
        torch.nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')   # kaiming initialization
