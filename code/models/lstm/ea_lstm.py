from configs.project_config import ProjectConfig
from configs.model_config.EALSTM_config import EALSTMConfig
from typing import Tuple
import torch
import torch.nn as nn


class EALSTM(nn.Module):
    """Implementation of the Entity-Aware-LSTM (EA-LSTM)
    TODO: Include paper ref and latex equations
    Parameters
    ----------
    input_size_dyn : int
        Number of dynamic features, which are those, passed to the LSTM at each time step.
    input_size_stat : int
        Number of static features, which are those that are used to modulate the input gate.
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self):
        super(EALSTM, self).__init__()

        self.input_size_dyn = EALSTMConfig.input_size_dyn
        self.input_size_stat = EALSTMConfig.input_size_stat
        self.hidden_size = EALSTMConfig.hidden_size
        self.output_size = EALSTMConfig.output_size
        self.batch_first = EALSTMConfig.batch_first
        self.initial_forget_bias = EALSTMConfig.initial_forget_bias
        self.seq_len = EALSTMConfig.seq_len
        self.pred_len = EALSTMConfig.pred_len
        self.linear1 = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
        self.linear2 = nn.Linear(in_features=self.seq_len, out_features=self.pred_len)

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(self.input_size_dyn, 3 * self.hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(self.hidden_size, 3 * self.hidden_size))
        self.weight_sh = nn.Parameter(torch.FloatTensor(self.input_size_stat, self.hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * self.hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, input_seq, seq_y) -> Tuple[torch.Tensor, torch.Tensor]:
        """[summary]
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor, containing a batch of sequences of the dynamic features. Shape has to match
            the format specified with batch_first.
        x_s : torch.Tensor
            Tensor, containing a batch of static features.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        x = torch.split(input_seq, [self.input_size_dyn-1, self.input_size_stat], dim=2)
        x_d = x[0]
        x_d = torch.cat([x_d, seq_y], dim=2)

        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        x_s = x[1].resize_([batch_size, 1, self.input_size_stat]).squeeze()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # calculate input gate only once because inputs are static
        bias_s_batch = (self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size()))
        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) +
                     torch.mm(x_d[t], self.weight_ih))
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        output = self.linear1(h_n)
        output = output.permute(0, 2, 1)
        output = self.linear2(output)
        output = output.permute(0, 2, 1)

        return output
