from ..data_shape_config import DataShapeConfig


class LSTMConfig:
    model_name = "LSTM"
    decode_mode = None
    input_size = DataShapeConfig.src_size
    hidden_size = 128
    num_layers = 2
    output_size = DataShapeConfig.tgt_size
    seq_len = DataShapeConfig.src_len
    pred_len = DataShapeConfig.pred_len

    model_info = f"{model_name}_{decode_mode}_{seq_len}_{pred_len}_[{input_size}-{output_size}-{hidden_size}-{num_layers}]"