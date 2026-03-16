import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        if top_k is not None and top_k > 0:
            top_values = mx.topk(logprobs, k=top_k, axis=-1)
            threshold = mx.min(top_values, axis=-1, keepdims=True)
            logprobs = mx.where(logprobs < threshold, -mx.inf, logprobs)
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = mx.take_along_axis(logprobs, sorted_idx, axis=-1)
            cumsum_logprobs = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
            mask = cumsum_logprobs < top_p
            mask[..., 0] = True
            logprobs[:, sorted_idx] = mx.where(mask, sorted_logprobs, -mx.inf)

        return mx.random.categorical(logprobs / temp, axis=-1)

    return sample
