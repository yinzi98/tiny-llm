import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable
from .kv_cache import TinyKvFullCache


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        logits = model(y.reshape(1, -1))[..., -1, :]
        logprobs = logits - mx.logsumexp(
            logits, keepdims=True
        )
        if sampler is None:
            return mx.argmax(logprobs, axis=-1)
        else:
            return sampler(logprobs)

    tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    token = None
    while True:
        token = _step(model, tokens)
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print (detokenizer.last_segment, end="", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        logits = model(y.reshape(1, -1), offset, kv_cache)[:, -1, :]
        logprobs = logits - mx.logsumexp(
            logits, keepdims=True
        )
        return mx.argmax(logprobs, axis=-1)
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
    offset = 0
    while True:
        token = _step(model, tokens, offset, kv_cache)
        mx.eval(token)
        if token.item() == tokenizer.eos_token_id:
            break
        tokens = mx.concat([tokens, token])
        offset += token.size
        detokenizer.add_token(token.item())
        print (detokenizer.last_segment, end="", flush=True)  



def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
