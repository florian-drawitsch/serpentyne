import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_input = nn.Conv3d(1, 3, (11, 11, 5))
        self.conv1 = nn.Conv3d(3, 3, (11, 11, 5))
        self.conv2 = nn.Conv3d(3, 3, (11, 11, 5))
        self.conv3 = nn.Conv3d(3, 3, (11, 11, 5))
        self.conv4 = nn.Conv3d(3, 3, (11, 11, 5))
        self.conv_output = nn.Conv3d(3, 2, (11, 11, 5))

    def forward(self, x):
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv_output(x)
        x = F.log_softmax(x, dim=1)
        return x
