import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    scores = mx.matmul(query, key.swapaxes(-2, -1))
    if scale is None:
        scale = mx.rsqrt(query.shape[-1])
    scores = scores * scale
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_size = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, E = query.shape
        projection_query = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_size)
            .transpose(0, 2, 1, 3)
        )
        projection_key = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_size)
            .transpose(0, 2, 1, 3)
        )
        projection_value = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_size)
            .transpose(0, 2, 1, 3)
        )
        attention = scaled_dot_product_attention_simple(
            projection_query,
            projection_key,
            projection_value,
            None,
            mask).transpose(0, 2, 1, 3).reshape(N, L, E)
        return linear(attention, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    H, S, _ = key.shape[-3:]
    HQ, L, D = query.shape[-3:]
    assert HQ % H == 0
    H_repeats = HQ // H
    B = query.shape[:-3]

    query = query.reshape(*B, H, H_repeats, L, D)
    key = key.reshape(*B, H, 1, S, D)
    value = value.reshape(*B, H, 1, S, D)

    if scale is None:
        scale = mx.rsqrt(key.shape[-1])
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        mask = mask.reshape(*B, H, H_repeats, L, S)
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value).reshape(*B, HQ, L, D)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
