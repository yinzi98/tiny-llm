import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights
from .kv_cache import TinyKvCache


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = dequantize_linear(wq)
        self.wk = dequantize_linear(wk)
        self.wv = dequantize_linear(wv)
        self.wo = dequantize_linear(wo)
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        if isinstance(offsets, int):
            rope_offsets = [slice(int(offsets), int(offsets + L))]
        else:
            rope_offsets = [slice(int(o), int(o + L)) for o in offsets]
        projection_query = self.rope(
            linear(x, self.wq, self.bq)
            .reshape(B, L, self.num_heads, self.head_dim),
            rope_offsets
        )
        projection_key = self.rope(
            linear(x, self.wk, self.bk)
            .reshape(B, L, self.num_kv_heads, self.head_dim),
            rope_offsets
        )
        projection_value = (
            linear(x, self.wv, self.bv)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
        )
        projection_query = projection_query.transpose(0, 2, 1, 3)
        projection_key = projection_key.transpose(0, 2, 1, 3)
        projection_value = projection_value.transpose(0, 2, 1, 3)
        projection_key, projection_value, _, mask = cache.update_and_fetch(
            projection_key, projection_value, L, mask
        )
        o = scaled_dot_product_attention_grouped(projection_query, projection_key, projection_value, None, mask)
        return linear(o.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size), self.wo)

class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = dequantize_linear(w_gate)
        self.w_up = dequantize_linear(w_up)
        self.w_down = dequantize_linear(w_down)

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
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
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
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x = x + self.mha(self.input_layernorm(x), offsets=offset, cache=cache, mask=mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        
        args = mlx_model.args
        self.layers = [
            Qwen2TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                theta=args.rope_theta,
                wq=layer.self_attn.q_proj,
                wk=layer.self_attn.k_proj,
                wv=layer.self_attn.v_proj,
                wo=layer.self_attn.o_proj,
                bq=layer.self_attn.q_proj.bias,
                bk=layer.self_attn.k_proj.bias,
                bv=layer.self_attn.v_proj.bias,
                w_gate=layer.mlp.gate_proj,
                w_up=layer.mlp.up_proj,
                w_down=layer.mlp.down_proj,
                w_input_layernorm=layer.input_layernorm.weight,
                w_post_attention_layernorm=layer.post_attention_layernorm.weight
            ) for layer in mlx_model.model.layers
        ]
        self.embedding = Embedding(args.vocab_size, args.hidden_size, dequantize_linear(mlx_model.model.embed_tokens))
        self.norm = RMSNorm(args.hidden_size, mlx_model.model.norm.weight, args.rms_norm_eps)
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.w_lm_head = mlx_model.lm_head

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        x = self.embedding(inputs)
        for idx in range(self.num_hidden_layers):
            x = self.layers[idx](x, offset, cache[idx], mask="causal")
        x = self.norm(x)
        if not self.tie_word_embeddings:
            x = linear(x, self.w_lm_head)
        else:
            x = self.embedding.as_linear(x)
        return x
