import torch
import torch.nn as nn
from configs.model_config.LSTM_config import LSTMConfig
from configs.project_config import ProjectConfig

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.device = ProjectConfig.device
        self.input_size = LSTMConfig.input_size
        self.hidden_size = LSTMConfig.hidden_size
        self.num_layers = LSTMConfig.num_layers
        self.output_size = LSTMConfig.output_size
        self.seq_len = LSTMConfig.seq_len
        self.pred_len = LSTMConfig.pred_len
        self.num_directions = 1     # 单向LSTM
        # self.batch_size = batch_size # paper中32 64 128
        # 定义LSTM模型，传入相关的参数
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bias=True, batch_first=True)
        # 两层全连接网络 <参数设置? >
        self.linear1 = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
        self.linear2 = nn.Linear(in_features=self.seq_len, out_features=self.pred_len)

    # 重写前向传播函数
    # 传入的inputs:input，(初始的隐状态h_0,初始的单元状态C_0)
    # 其中的input: input(seq_len,batch_size,input_size)
    def forward(self, input_seq, seq_y):
        batch_size = input_seq.shape[0]
        # h_0C_0可以随机初始化
        h_0 = torch.rand(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.rand(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # output = (batch_sizeseq_len,num_direction*hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        output = self.linear1(output)
        output = output.permute(0, 2, 1)
        output = self.linear2(output)
        output = output.permute(0, 2, 1)
        # 确定输出的信息
        return output