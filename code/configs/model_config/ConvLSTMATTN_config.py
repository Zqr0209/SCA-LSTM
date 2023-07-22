from ..data_shape_config import DataShapeConfig


class ConvLSTMATTNConfig:
    model_name = "ConvLSTMATTN"
    decode_mode = None
    hidden_size = 128
    seq_len = DataShapeConfig.src_len
    output_len = DataShapeConfig.pred_len
    input_size = hidden_size
    n_heads = 8
    dropout_rate = 0.2
    kernel_size = 3
    in_channels = DataShapeConfig.src_size
    out_channels = hidden_size
    dilation = 1

    model_info = f"{model_name}_{decode_mode}_[[{seq_len}-{output_len}-{input_size}-{hidden_size}]" \
                 f"[*-{output_len}-{input_size}-{hidden_size}]-{dropout_rate}]"