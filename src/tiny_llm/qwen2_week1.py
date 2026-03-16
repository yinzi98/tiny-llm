import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        projection_query = self.rope(
            linear(x, self.wq, self.bq)
            .reshape(B, L, self.num_heads, self.head_dim)
        )
        projection_key = self.rope(
            linear(x, self.wk, self.bk)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
        )
        projection_value = (
            linear(x, self.wv, self.bv)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
        )
        projection_query = projection_query.transpose(0, 2, 1, 3)
        projection_key = projection_key.transpose(0, 2, 1, 3)
        projection_value = projection_value.transpose(0, 2, 1, 3)
        o = scaled_dot_product_attention_grouped(projection_query, projection_key, projection_value, None, mask)
        return linear(o.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size), self.wo)

class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)

class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.input_layernorm = RMSNorm(
            hidden_size,
            w_input_layernorm,
            rms_norm_eps)

        self.mha = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq,
            wk,
            wv,
            wo,
            bq,
            bk,
            bv,
            max_seq_len,
            theta)

        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            w_post_attention_layernorm,
            rms_norm_eps)
        
        self.mlp = Qwen2MLP(
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x = x + self.mha(self.input_layernorm(x), mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        args = mlx_model.args
        self.layers = [
            Qwen2TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                theta=args.rope_theta,
                wq=dequantize_linear(layer.self_attn.q_proj),
                wk=dequantize_linear(layer.self_attn.k_proj),
                wv=dequantize_linear(layer.self_attn.v_proj),
                wo=dequantize_linear(layer.self_attn.o_proj),
                bq=layer.self_attn.q_proj.bias,
                bk=layer.self_attn.k_proj.bias,
                bv=layer.self_attn.v_proj.bias,
                w_gate=dequantize_linear(layer.mlp.gate_proj),
                w_up=dequantize_linear(layer.mlp.up_proj),
                w_down=dequantize_linear(layer.mlp.down_proj),
                w_input_layernorm=layer.input_layernorm.weight,
                w_post_attention_layernorm=layer.post_attention_layernorm.weight
            ) for layer in mlx_model.model.layers
        ]
        self.embedding = Embedding(args.vocab_size, args.hidden_size, dequantize_linear(mlx_model.model.embed_tokens))
        self.norm = RMSNorm(args.hidden_size, mlx_model.model.norm.weight, args.rms_norm_eps)
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        x = self.embedding(inputs)
        for layer in self.layers:
            x = layer(x, mask="causal")
        x = self.norm(x)
        if not self.tie_word_embeddings:
            x = linear(x, self.w_lm_head)
        else:
            x = self.embedding.as_linear(x)
        return x