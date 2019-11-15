import datetime
from pathlib import Path
from serpentyne.data.dataset import D3d
# from serpentyne.net.segem import Net
from serpentyne.net.minimal import Net
from serpentyne.train.train import Train
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


data_path = Path("/home/drawitschf/Code/mpi/florian/myelinPrediction/data/ex145_myelin.hdf5")

input_shape = [75, 75, 31]
output_shape = [15, 15, 7]
# input_shape = [201, 201, 31]
# output_shape = [141, 141, 7]
# input_shape = [101, 101, 31]
# output_shape = [41, 41, 7]
input_shape = [91, 91, 31]
# output_shape = [91, 91, 7]
output_shape = [77, 77, 21]
pad_target = False

trainset = D3d(
    data_path=data_path,
    input_shape=input_shape,
    output_shape=output_shape,
    pad_target=pad_target)

# inds, counts = trainset.get_item_inds_with_labels()
# inds = np.asarray(inds)
# counts = np.asarray(counts)
# counts_sorted_inds = np.argsort(counts)
# inds_sorted = inds[counts_sorted_inds]
# inds_sorted[0:-20]

#1650
# 987, 971, 932, 943, 832, 957, 942, 941, 955, 956
# trainset.show_sample(943)

batch_size = 8
shuffle = True
num_workers = 0
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

net = Net()

class_weights = torch.tensor([0.01, 0.99])
criterion = nn.NLLLoss(class_weights)
optimizer = optim.Adam(net.parameters(), lr=0.001)
num_epochs = 30
timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

run_path = f'/home/drawitschf/Code/private/serpentyne/runs/myelin_segem_v01_{timestamp}'

cuda = False
cuda_dev_idx = 1


train = Train(dataloader, net, optimizer, criterion, num_epochs, run_path, cuda, cuda_dev_idx)
train.run()



