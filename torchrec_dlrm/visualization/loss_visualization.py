import numpy as np
from matplotlib import pyplot as plt
from numpy import loadtxt
# load array
loss_data_path = "/N/u/haofeng/BigRed200/new_dlrm/torchrec_dlrm/loss_result.csv"
acc_data_path = "/N/u/haofeng/BigRed200/new_dlrm/torchrec_dlrm/acc_result.csv"
loss_data = loadtxt(loss_data_path, delimiter=',')
acc_data = loadtxt(acc_data_path, delimiter=',')
# print the array
print(loss_data)

x = np.array(range(len(loss_data)))
print(x)
plt.plot(x,loss_data,'o-',color='r',label="loss")
plt.savefig("loss.png")