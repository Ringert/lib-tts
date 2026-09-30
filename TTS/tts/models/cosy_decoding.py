"""Instance-local bounded decoding for short COSY markup segments."""

from contextlib import contextmanager
from dataclasses import dataclass


class SegmentTokenLimitError(RuntimeError):
    """A markup segment did not finish normally within its token budget."""


@dataclass
class _DecodeStatus:
    original_maximum: int
    maximum: int
    count: int = 0
    completed: bool = False


class MarkupDecoder:
    """Shared with the backend worker thread; caller holds the adapter lock.

    In the pinned CPU decoder, normal completion before the iteration limit
    means an accepted stop token. Count before downstream silent-token filtering.
    Do not raise exhaustion errors in the upstream producer thread: consume and
    clean up the backend first, then reject the result in the caller.
    """

    def __init__(self, llm):
        self.active = None
        original = llm.inference_wrapper

        def inference(lm_input, sampling, min_len, max_len, uuid):
            if self.active is None:
                yield from original(lm_input, sampling, min_len, max_len, uuid)
                return
            status = _DecodeStatus(max_len, max(max_len, 75))
            self.active.append(status)
            for token in original(lm_input, sampling, min_len, status.maximum, uuid):
                status.count += 1
                yield token
            status.completed = True

        llm.inference_wrapper = inference

    @contextmanager
    def segment(self):
        self.active = []
        try:
            yield
            if not self.active or any(
                not s.completed or s.count >= s.maximum for s in self.active
            ):
                raise SegmentTokenLimitError("Speech segment reached its token limit.")
        finally:
            self.active = None
