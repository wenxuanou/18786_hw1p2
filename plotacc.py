import numpy as np
import matplotlib.pyplot as plt

train_acc_path = "log/train_acc.npy"
val_acc_path = "log/val_acc.npy"

train_acc = np.load(train_acc_path)
val_acc = np.load(val_acc_path)

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



plt.show()