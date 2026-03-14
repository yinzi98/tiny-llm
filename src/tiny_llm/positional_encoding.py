import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0
        self.half_dims = dims // 2
        inner = mx.arange(0, self.half_dims, dtype=mx.float32) / self.half_dims
        freqs = mx.power(base, -inner)
        t = mx.arange(seq_len)
        freqs = mx.outer(t, freqs)
        self.sin_freqs = mx.sin(freqs)
        self.cos_freqs = mx.cos(freqs)
        self.base = base
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape

        if offset is not None:
            if isinstance(offset, list):
                assert len(offset) == L
                for o in offset:
                    assert o.stop - o.start == L
                offset = mx.array(
                    [list(range(i.start, i.stop)) for i in offset]
                )
            elif isinstance(offset, slice):
                assert offset.stop - offset.start == L
        sin_basis = (
            self.sin_freqs[:L, :] if offset is None else self.sin_freqs[offset, :]
        ).reshape(-1, L, 1, self.half_dims)
        cos_basis = (
            self.cos_freqs[:L, :] if offset is None else self.cos_freqs[offset, :]
        ).reshape(-1, L, 1, self.half_dims)
        if self.traditional:
            x = x.reshape(N, L, H, self.half_dims, 2)
            x0 = x[..., 0]
            x1 = x[..., 1]
        else:
            x0 = x[..., : self.half_dims]
            x1 = x[..., self.half_dims :]
        ox = x0 * cos_basis - x1 * sin_basis
        oy = x0 * sin_basis + x1 * cos_basis

        if self.traditional:
            o = mx.stack([ox, oy], axis = -1)
            o = o.reshape(N, L, H, D)
        else:
            o = mx.concat([ox, oy], axis = -1)
        return o
