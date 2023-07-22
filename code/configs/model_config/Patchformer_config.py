from ..data_shape_config import DataShapeConfig


class PatchformerConfig:
    model_name = "Patchformer"
    decode_mode = "NAR"  # NAR or AR
    d_model = 64  # embedding size, d_model matters a lot
    n_heads = 4  # number of heads in multi-head attention
    assert d_model % n_heads == 0
    d_head = d_model // n_heads  # dimension of each head in multi-head attention
    n_encoder_layers = 4  # number of encoder layer
    n_decoder_layers = 4  # number of decoder layer
    assert n_encoder_layers == n_decoder_layers  # we assert they are equal here
    d_ff = 256  # feedforward dimension
    dropout_rate = 0.1  # dropout rate
    patch_len = 30   # patch_len为分片长度
    stride = 30  # stride为相邻两个分片不重复的长度，stride = patch_len表示相邻两个分片不重复且紧挨
    src_len = DataShapeConfig.src_len
    tgt_len = DataShapeConfig.tgt_len
    pred_len = DataShapeConfig.pred_len
    src_size = DataShapeConfig.src_size
    tgt_size = DataShapeConfig.tgt_size
    model_info = f"{model_name}_{decode_mode}_[{d_model}-{n_heads}-{patch_len}-{stride}-{n_encoder_layers}-{d_ff}-{dropout_rate}]"