import torch
import torch.utils.data as data

import numpy as np

class MyDataset(data.Dataset):
    def __init__(self, X, Y, offset=1, context=1):

        ### Add data and label to self (1-2 lines)
        self.X = X
        self.Y = Y

        ### Define data index mapping (4-6 lines)
        index_map_X = []
        for i, x in enumerate(X):
            for j, xx in enumerate(x):
                index_pair_X = (i, j)
                index_map_X.append(index_pair_X)

        ### Define label index mapping (4-6 lines)
        index_map_Y = []
        for i, y in enumerate(Y):
            for j, yy in enumerate(y):
                index_pair_Y = (i, j)
                index_map_Y.append(index_pair_Y)

        ### Assert the data index mapping and label index mapping are the same (1 line)
        assert (set(index_map_X) == set(index_map_Y))

        ### Assign data index mapping to self (1 line)
        self.index_map = index_map_X

        ### Add length to self (1 line)
        self.length = len(index_map_X)

        ### Add context and offset to self (1-2 line)
        self.offset = offset
        self.context = context

        ### Zero pad data as-needed for context size = 1 (1-2 lines)
        for i, x in enumerate(self.X):
            self.X[i] = np.pad(x,
                               ((self.offset, offset), (0, 0)),
                               'constant',
                               constant_values=0)

    def __len__(self):

        ### Return length (1 line)
        return self.length

    def __getitem__(self, index):

        ### Get index pair from index map (1-2 lines)
        i, j = self.index_map[index]

        ### Calculate starting timestep using offset and context (1 line)
        start = j + self.offset - self.context

        ## Calculate ending timestep using offset and context (1 line)
        end = j + self.offset + self.context + 1

        ### Get data at index pair with context (1 line)
        x = self.X[i][start:end, :]

        ### Get label at index pair (1 line)
        y = self.Y[i][j]

        ### Return data at index pair with context and label at index pair (1 line)
        return x, y

    def collate_fn(batch):

        ### Select all data from batch (1 line)
        batch_x = [x for x, y in batch]

        ### Select all labels from batch (1 line)
        batch_y = [y for x, y in batch]

        ### Convert batched data and labels to tensors (2 lines)
        batch_x = torch.as_tensor(batch_x)
        batch_y = torch.as_tensor(batch_y)

        ### Return batched data and labels (1 line)
        return batch_x, batch_y