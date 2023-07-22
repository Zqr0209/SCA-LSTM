from ..data_shape_config import DataShapeConfig


class EALSTMConfig:
    model_name = "EALSTM"
    decode_mode = "NAR"
    input_size_dyn = DataShapeConfig.dynamic_size + 1   # 气象属性 + runoff
    input_size_stat = DataShapeConfig.static_size   # 静态属性
    hidden_size = 256
    output_size = DataShapeConfig.tgt_size
    seq_len = DataShapeConfig.src_len
    pred_len = DataShapeConfig.pred_len
    batch_first = True
    initial_forget_bias = 0

    model_info = f"{model_name}_{seq_len}_{pred_len}_[{input_size_dyn}-{input_size_stat}-{hidden_size}]"