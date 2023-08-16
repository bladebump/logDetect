from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder
import torch
from torch.nn import Embedding,AdaptiveAvgPool1d
from .base import BaseClassifier

def get_retnet_model(vocab_size,EmbeddingLayer,layer_num=12):
    config = RetNetConfig(
        decoder_embed_dim=768,
        decoder_retention_heads=3,
        decoder_ffn_embed_dim=1536,
        decoder_layers=layer_num,
        decoder_normalize_before=True,
        activation_fn='gelu',
        dropout=0.0,
        drop_path_rate=0.0,
        activation_dropout=0.0,
        no_scale_embedding=True,
        layernorm_embedding=False,
        moe_freq=0,
        moe_top1_expert=False,
        moe_expert_count=0,
        moe_gating_use_fp32=True,
        moe_eval_capacity_token_fraction=0.25,
        moe_second_expert_policy='random',
        moe_normalize_gate_prob_before_dropping=False,
        use_xmoe=False,
        rel_pos_buckets=0,
        max_rel_pos=0,
        deepnorm=False,
        subln=True,
        multiway=False,
        share_decoder_input_output_embed=False,
        max_target_positions=1024,
        no_output_layer=False,
        layernorm_eps=1e-5,
        chunkwise_recurrent=False,
        recurrent_chunk_size=512,
        vocab_size=vocab_size,
        checkpoint_activations=False,
        fsdp=False,
        ddp_rank=0,
        xpos_rel_pos=False,
        xpos_scale_base=512,
        )
    model = RetNetDecoder(config,EmbeddingLayer)

    return model

class RetNetClassifier(BaseClassifier):
    def __init__(self, vocab_size, num_classes, lr,layer_num=1):
        super(RetNetClassifier, self).__init__(vocab_size, num_classes, lr)

        self.embed = Embedding(vocab_size, 768)
        self.model = get_retnet_model(vocab_size,self.embed,layer_num=layer_num)
        self.gblobel_avg_pool = AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(vocab_size, num_classes)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.model(input_ids)
        x = self.classifier(x[0])
        x = x.permute(0, 2, 1)
        x = self.gblobel_avg_pool(x)
        return x.squeeze()


if __name__ == "__main__":
    model = RetNetClassifier(vocab_size=1000,num_classes=6,lr=1e-3,layer_num=1)

    # random input
    x = torch.randint(0, 1000, (2, 512))
    y = model(x)
    print(y.shape)