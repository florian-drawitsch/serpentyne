import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt


class Train:

    def __init__(self,
                 dataloader: DataLoader,
                 net,
                 optimizer,
                 criterion,
                 num_epochs,
                 run_path,
                 cuda=False,
                 cuda_dev_idx = 0):

        self.dataloader = dataloader
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.run_path = run_path
        self.cuda = cuda

        if cuda is True & torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(cuda_dev_idx))
            self.net.to(self.device)
        else:
            self.device = torch.device("cpu")

    def run(self):
        print('Starting training')

        writer = SummaryWriter(self.run_path)

        for epoch in range(self.num_epochs):

            running_loss = 0.0
            for i, data in enumerate(self.dataloader):
                # get the inputs
                inputs, labels = data
                if self.cuda is True:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                # inputs, labels = self.dataloader.dataset[544]
                # inputs = inputs.unsqueeze(0)
                # labels = labels.unsqueeze(0)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                labels = torch.squeeze(labels, dim=1)
                labels = labels.type(torch.LongTensor)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                print('loss: {}'.format(loss.item()))

                running_loss += loss.item()
                if i % 10 == 0:  # every 10 mini-batches

                    # write to stout
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

                    # write to tensorboard
                    writer.add_scalar('loss', loss.item(), i)

                    inputs_tbx = Train.inputs2tbx(inputs)
                    writer.add_figure('inputs', inputs_tbx, i)

                    outputs_tbx = Train.outputs2tbx(outputs)
                    writer.add_figure('outputs', outputs_tbx, i)

                    labels_tbx = Train.labels2tbx(labels)
                    writer.add_figure('labels', labels_tbx, i)

                    # save model weights
                    torch.save(self.net.state_dict(), Path(self.run_path).joinpath('model_state'))


    @staticmethod
    def inputs2tbx(inputs):

        inputs = inputs.detach().numpy()[0, 0, :, :, np.floor(inputs.shape[4] / 2).astype(int)]

        fig, ax = plt.subplots()
        ax = ax.imshow(inputs, cmap='gray', vmin=-3, vmax=3)  # , cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(ax)

        return fig

    @staticmethod
    def outputs2tbx(outputs):

        outputs = outputs.detach().numpy()[0, 1, :, :, np.floor(outputs.shape[4] / 2).astype(int)]
        outputs = np.exp(outputs)

        fig, ax = plt.subplots()
        ax = ax.imshow(outputs, vmin=0, vmax=1)  # , cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(ax)

        return fig


    @staticmethod
    def labels2tbx(labels):

        labels = labels.detach().numpy()[0, :, :, np.floor(labels.shape[3] / 2).astype(int)]

        fig, ax = plt.subplots()
        ax = ax.imshow(labels)
        fig.colorbar(ax)

        return fig




