from ..data_shape_config import DataShapeConfig


class LSTMATTNConfig:
    model_name = "LSTMATTN"
    decode_mode = None
    seq_len = DataShapeConfig.src_len
    output_len = DataShapeConfig.pred_len
    input_size = DataShapeConfig.src_size
    n_heads = 8
    hidden_size = 128
    dropout_rate = 0.2

    model_info = f"{model_name}_{decode_mode}_[[{seq_len}-{output_len}-{input_size}-{hidden_size}]" \
                 f"[*-{output_len}-{input_size}-{hidden_size}]-{dropout_rate}]"