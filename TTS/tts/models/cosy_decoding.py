"""Instance-local bounded decoding for short COSY markup segments."""

from contextlib import contextmanager
from dataclasses import dataclass


def preserve_minimum_tokens(llm):
    """Mask CV3's complete stop set before its existing sampling decision.

    The pinned inherited sampler masks only speech_token_size (CV3's SOS).
    Keep the correction local to this model and preserve the original RAS path.
    """
    stop_ids = tuple(llm.stop_token_ids)
    width = llm.llm_decoder.out_features
    if not stop_ids or any(
        type(token) is not int or not 0 <= token < width for token in stop_ids
    ):
        raise ValueError("Invalid CosyVoice stop-token configuration")
    original = llm.sampling_ids

    def sampling_ids(weighted_scores, decoded_tokens, sampling, ignore_eos=True):
        if ignore_eos:
            if weighted_scores.ndim != 1 or weighted_scores.shape[0] != width:
                raise ValueError("Unexpected CosyVoice sampling score shape")
            weighted_scores = weighted_scores.clone()
            weighted_scores[list(stop_ids)] = -float("inf")
        return original(weighted_scores, decoded_tokens, sampling, ignore_eos)

    llm.sampling_ids = sampling_ids


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
