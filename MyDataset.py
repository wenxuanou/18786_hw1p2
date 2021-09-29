import torch
import torch.utils.data as data

import numpy as np

class MyDataset(data.Dataset):
    # borrow from previous homework

    def __init__(self, X, Y, offset=1, context=1):

        ### Add data and label to self (1-2 lines)
        self.X = X            # X: (datafile_num, (time_frame, frequency))
        self.Y = Y            # Y: (datafile_num, (time_frame, ))


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

        ### Zero pad data for context
        for i, x in enumerate(self.X):
            self.X[i] = np.pad(x,
                               ((self.context, self.context), (0, 0)),
                               'constant',
                               constant_values=0)   # pad with zeros, x is 2D
            
            self.X[i] = self.X[i].astype(np.float32)
            
        for i, y in enumerate(self.Y):
            self.Y[i] = self.Y[i].astype(np.int64)
            
            
    def __len__(self):

        ### Return length (1 line)
        return self.length

    def __getitem__(self, index):

        ### Get index pair from index map (1-2 lines)
        i, j = self.index_map[index]

        ### Calculate starting timestep using offset and context (1 line)
        start = j + self.offset - self.context

        ## Calculate ending timestep using offset and context (1 line)
        end = start + 2 * self.context + 1

        assert start < end and self.X[i].size > 0

        ### Get data at index pair with context (1 line)
        x = self.X[i][start:end, :]

        ### Get label at index pair (1 line)
        y = self.Y[i][j]

        ### Return data at index pair with context and label at index pair (1 line)
        return x, y

    def collate_fn(batch):

        batch_x = []
        batch_y = []
        for x, y in batch:
            ### Select all data from batch (1 line)
            batch_x += [x]

            ### Select all labels from batch (1 line)
            batch_y += [y]

        ### Convert batched data and labels to tensors (2 lines)
        batch_x = torch.as_tensor(batch_x)
        batch_y = torch.as_tensor(batch_y)

        ### Return batched data and labels (1 line)
        return batch_x, batch_y