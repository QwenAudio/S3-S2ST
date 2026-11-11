"""Backend-neutral 640-ms streaming controller.

The controller adds no special vocabulary tokens.  WRITE opens a normal
assistant ChatML turn; generation stops on the existing ``<|im_end|>`` and the
next audio block opens another normal user turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


IM_END = "<|im_end|>"


class Backend(Protocol):
    def score_after_audio(self, pcm_640ms: bytes, history: list[dict]) -> float: ...
    def generate(self, history: list[dict], stop: str = IM_END) -> str: ...


@dataclass
class StreamingController:
    backend: Backend
    threshold: float = 0.5
    min_read_blocks: int = 1
    max_read_blocks: int = 20
    history: list[dict] = field(default_factory=list)
    pending_audio: list[bytes] = field(default_factory=list)

    def push(self, pcm_640ms: bytes) -> dict:
        self.pending_audio.append(pcm_640ms)
        p_write = self.backend.score_after_audio(pcm_640ms, self.history)
        forced = len(self.pending_audio) >= self.max_read_blocks
        write = len(self.pending_audio) >= self.min_read_blocks and (p_write >= self.threshold or forced)
        result = {"decision": "WRITE" if write else "READ", "p_write": p_write, "forced": forced}
        if not write:
            return result
        self.history.append({"role": "user", "audio_blocks": self.pending_audio.copy()})
        self.pending_audio.clear()
        text = self.backend.generate(self.history, stop=IM_END)
        self.history.append({"role": "assistant", "content": text, "terminated_by": IM_END})
        result["text"] = text
        return result

    def finish(self, final_pcm: bytes = b"") -> dict:
        """Force the residual (<640 ms) audio to flush at end of stream."""
        if final_pcm:
            self.pending_audio.append(final_pcm)
        if not self.pending_audio:
            return {"decision": "END", "text": ""}
        self.history.append({"role": "user", "audio_blocks": self.pending_audio.copy(), "end_of_stream": True})
        self.pending_audio.clear()
        text = self.backend.generate(self.history, stop=IM_END)
        self.history.append({"role": "assistant", "content": text, "terminated_by": IM_END})
        return {"decision": "WRITE", "p_write": 1.0, "forced": True, "end_of_stream": True, "text": text}


def vllm_note() -> str:
    return (
        "Current offline vLLM treats <|im_end|> as request-finished. The functional "
        "MVP therefore resubmits accumulated interleaved history with prefix caching; "
        "a later engine patch can retain the same KV cache across turns."
    )
