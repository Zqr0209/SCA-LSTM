# 只有一个encoder和一个decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config.ConvLSTMATTN_config import ConvLSTMATTNConfig
from configs.project_config import ProjectConfig

'''
Encoder1：全部的气象信息
Encoder2：径流真实值
Decoder：两个编7天结果拼接
'''


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask, d_head):
        """
        Q: [batch_size, n_heads, len_q, d_head)]
        K: [batch_size, n_heads, len_k(=len_v), d_head]
        V: [batch_size, n_heads, len_v(=len_k), d_head]
        mask: [batch_size, n_heads, seq_len, seq_len]
        """
        # transpose将-1维和-2维交换
        # len_q: target len, len_k: source len
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(d_head, dtype=torch.float32))  # scores : [batch_size, n_heads, len_q, len_k]
        # 最后两维len_q和len_k分别表示target和source。

        if mask is not None:
            scores += mask

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_head]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, input_Q, input_K, input_V, mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k(=len_v), d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        batch_size = input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        # Q: [batch_size, n_heads, len_q, d_head]
        # K: [batch_size, n_heads, len_k(=len_v), d_head]
        # V: [batch_size, n_heads, len_v(=len_k), d_head]

        # mask: [seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, mask, self.d_head)
        # context: [batch_size, n_heads, len_q, d_head], attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_head)
        # context: [batch_size, len_q, n_heads * d_head]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return output, attn  # 返回MultiHeadAttention的output和attn矩阵，attn矩阵用于可视化

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

# 对x_1~x_{n+m}进行编码
class EncoderRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = ConvLSTMATTNConfig.seq_len
        self.hidden_size = ConvLSTMATTNConfig.hidden_size
        self.input_size = ConvLSTMATTNConfig.input_size
        self.output_len = ConvLSTMATTNConfig.output_len

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, bias=True,
                            dropout=0, batch_first=True, bidirectional=False)

    def forward(self, inputs):
        output, (h_n, c_n) = self.lstm(inputs)  # output.shape = [512, 22, 50]

        return output, (h_n, c_n)


class ConvLSTMATTN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = ProjectConfig.device
        self.encoder_obs = EncoderRNN()

        self.output_len = ConvLSTMATTNConfig.output_len
        self.n_heads = ConvLSTMATTNConfig.n_heads
        self.convlayer = CausalConv1d(ConvLSTMATTNConfig.in_channels, ConvLSTMATTNConfig.out_channels, ConvLSTMATTNConfig.kernel_size, dilation=ConvLSTMATTNConfig.dilation)
        self.attn_layer = MultiHeadAttention(ConvLSTMATTNConfig.hidden_size, self.n_heads)
        self.cat_linear = nn.Linear(in_features=ConvLSTMATTNConfig.hidden_size + ConvLSTMATTNConfig.hidden_size,
                                    out_features=ConvLSTMATTNConfig.input_size,
                                    bias=True)
        self.input_project = nn.Linear(in_features=1,
                                       out_features=ConvLSTMATTNConfig.input_size,
                                       bias=False)

        self.out = nn.Linear(in_features=ConvLSTMATTNConfig.input_size, out_features=1, bias=False)
        self.dropout = nn.Dropout(p=ConvLSTMATTNConfig.dropout_rate)



    def forward(self, src, tgt):

        src = src.permute(0, 2, 1)
        src = self.convlayer(src)
        encoder_input = src.permute(0, 2, 1)
        encoder_obs_outputs, (h_enc, c_enc) = self.encoder_obs.forward(encoder_input)

        batch_size = src.shape[0]
        before_len = tgt.shape[1]
        # hidden_size_dec = LSTMATTNS2SConfig.hidden_size_d

        output_len = self.output_len
        outputs = torch.empty((batch_size, output_len, 1)).to(self.device)


        for i in range(output_len):
            h_n_T = h_enc.transpose(0, 1)
            Q = h_n_T
            K = encoder_obs_outputs[:, :before_len + i + 1, :]
            attn_value, _ = self.attn_layer(Q, K, K, None)
            h_n_after = self.cat_linear(torch.cat((h_n_T, attn_value), dim=2))
            h_n_after = self.dropout(h_n_after)
            outputs = self.out(h_n_after)


        # outputs = outputs.squeeze(-1)
        # outputs.shape: (512，7)

        return outputs



