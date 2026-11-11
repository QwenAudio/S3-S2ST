try:
    from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
except Exception as exc:
    raise ImportError("rotary_embedding_torch is required for the local x_transformers compatibility shim") from exc

def apply_rotary_pos_emb(t, freqs, scale=1.0):
    return apply_rotary_emb(freqs, t, scale=scale)

__all__ = ["RotaryEmbedding", "apply_rotary_pos_emb"]
