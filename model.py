import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet
import IPython as IP

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout): # 1, 10, [25, 25 ... 25], 7.
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # print('Ready to linearize')
        # IP.embed()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        # print('Forward pass of TCN')
        # IP.embed()
        y1 = self.tcn(inputs)  # input should have dimension (batch, channels, seq_length)
        print('Final Step TCN forward')
        IP.embed()
        o = self.linear(y1[:, :, -1])
        return self.relu(o) # Maybe Relu helps to discard negative EMG values. Otherwise return o
        # return F.Linear(o, dim=1)
