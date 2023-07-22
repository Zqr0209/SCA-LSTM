from ..data_shape_config import DataShapeConfig


class TCNConfig:
    model_name = "TCN"
    decode_mode = "NAR"
    num_channels = [64, 64, 64, 64, 1]
    kernel_size = 3
    stride = 1
    dilation = 1
    dropout = 0.7  # dropout rate
    src_len = DataShapeConfig.src_len
    tgt_len = DataShapeConfig.tgt_len
    pred_len = DataShapeConfig.pred_len
    src_size = DataShapeConfig.src_size
    tgt_size = DataShapeConfig.tgt_size
    model_info = f"{model_name}_[{kernel_size}-{stride}-{dilation}-{dropout}]"
