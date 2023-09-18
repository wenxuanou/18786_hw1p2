import numpy as np

testdata_path = "data/toy_test_data.npy"
data = np.load(testdata_path, allow_pickle=True)

data = data[0]

print(data.shape)

for i, x in enumerate(data):
    assert x.size > 0

