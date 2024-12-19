import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Reading the Loss Values of Each Epoch from File
f = open('loss_values.txt','rb')
x = pickle.load(f)
f.close()

y = torch.tensor(x)
y1 = y.numpy()
n = torch.arange(1,201,1)
n1 = n.numpy()

plt.plot(n1,y1, color='red',linewidth=2.5)

plt.yticks(np.arange(min(y1),max(y1),0.1))
plt.xticks(np.arange(1,200,5))

plt.grid(True,color='black',linewidth=0.3)
plt.xlabel('Number of Epoch', fontsize=16)
plt.ylabel('Cross Entropy Loss', fontsize=16)
plt.suptitle('Loss Plot of Vision Transformer', fontsize=20)
plt.show()


