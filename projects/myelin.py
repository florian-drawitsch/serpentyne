import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from serpentyne.data import dataset
from serpentyne.net import segem
from serpentyne.train import train

data_source_dir = "/home/drawitschf/Code/mpi/florian/myelinPrediction/data/"
input_shape = [71, 71, 35]
output_shape = [11, 11, 11]
data_format = 'mat'
data_include_idx = [1]
pad_target = False
batch_size = 1
shuffle = False
num_workers = 1
num_epochs = 1

dataset = dataset.D3d(
    data_source_dir=data_source_dir,
    input_shape=input_shape,
    output_shape=output_shape,
    data_include_idx=data_include_idx,
    pad_target=pad_target)
dataset = dataset.normalize_data_auto()

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

net = segem.Net()

class_weights = torch.tensor([0.1, 1.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainer = train.Train(dataloader,
                      net,
                      batch_size,
                      shuffle,
                      num_workers,
                      optimizer,
                      criterion,
                      class_weights,
                      num_epochs)
trainer.run()
