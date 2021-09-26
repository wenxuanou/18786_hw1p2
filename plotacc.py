import numpy as np
import matplotlib.pyplot as plt

train_acc_path = "log/train_acc.npy"
val_acc_path = "log/val_acc.npy"

train_acc = np.load(train_acc_path)
val_acc = np.load(val_acc_path)

plt.figure(1)
plt.plot(train_acc)

plt.figure(2)
plt.plot(val_acc)

plt.show()