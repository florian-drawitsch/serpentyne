import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_input = nn.Conv3d(1, 16, (7, 7, 3))
        self.conv_1 = nn.Conv3d(16, 16, 3)
        self.conv_2 = nn.Conv3d(16, 16, 3)
        self.conv_3 = nn.Conv3d(16, 16, 3)
        self.conv_output = nn.Conv3d(16, 2, 3)

    def forward(self, x):
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = self.conv_output(x)
        x = F.log_softmax(x, dim=1)
        return x
