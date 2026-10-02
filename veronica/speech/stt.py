import asyncio
from typing import NamedTuple


import numpy as np
from faster_whisper import WhisperModel

# See transcribe_scored: bounded so noise can't hold up a turn for minutes.
DECODE_TEMPERATURES = (0.0, 0.4)
MAX_NEW_TOKENS = 160


def stt_spec(settings, mode: str) -> tuple[str, str | None, str]:
    """Which whisper models (main, partial) and language hint a language
    mode uses: "en" -> the English-only pair with language="en"; "hi" ->
    the multilingual pair pinned to Hindi; "auto" -> the multilingual pair
    with language=None so whisper detects per utterance. Shared by
    __main__ (startup) and the orchestrator (a "speak hindi" switch)."""
    if mode == "en":
        return settings.whisper_model, "en", settings.partial_stt_model
    language = "hi" if mode == "hi" else None
    return settings.whisper_multilingual_model, language, settings.partial_stt_multilingual_model


def expected_langs(mode: str) -> tuple[str, ...]:
    """The languages the user speaks in a language mode. Whisper often
    labels spoken Hindi as Urdu, so that counts as Hindi."""
    return ("en",) if mode == "en" else ("en", "hi", "ur")


class Transcript(NamedTuple):
    """A transcript with whisper's own opinion of it. The scores are None
    when the model gave none (no segments, or a test double): unscored
    text is taken at its word."""
    text: str
    language: str
    language_probability: float | None = None
    no_speech_prob: float | None = None     # the worst 30 s window
    avg_logprob: float | None = None        # token-weighted over segments
    compression_ratio: float | None = None  # the worst segment; high = a loop


def _scored(segments, info, pinned: str | None) -> Transcript:
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    language = pinned or getattr(info, "language", None) or "en"
    lang_p = getattr(info, "language_probability", None)
    scored = [s for s in segs if getattr(s, "avg_logprob", None) is not None]
    if not scored:
        return Transcript(text, language, lang_p)
    weights = [max(1, len(getattr(s, "tokens", None) or ())) for s in scored]
    return Transcript(
        text, language, lang_p,
        no_speech_prob=max(s.no_speech_prob for s in scored),
        avg_logprob=sum(s.avg_logprob * w for s, w in zip(scored, weights)) / sum(weights),
        compression_ratio=max(s.compression_ratio for s in scored),
    )


class Transcriber:
    """faster-whisper wrapper. Input: int16 mono 16 kHz."""

    _model_cls = WhisperModel  # swapped in tests

    def __init__(self, model_name: str, language: str | None = "en") -> None:
        self.model_name = model_name
        self.language = language          # None = let whisper detect
        self._model = self._model_cls(model_name, device="cpu", compute_type="int8")

    def set_language(self, language: str | None) -> None:
        self.language = language

    def transcribe_scored(self, pcm16: np.ndarray) -> Transcript:
        audio = pcm16.astype(np.float32) / 32768.0
        segments, info = self._model.transcribe(
            audio, beam_size=1, language=self.language, vad_filter=False,
            # Bounded decoding. On noise, whisper's default temperature
            # fallback (0.0→1.0 in six steps) retries a hallucination loop
            # again and again: a 3 s pink-noise clip took 297 s, long enough
            # to stall a confirm and the task waiting on it. Spoken turns are
            # short, so one retry and a token cap lose nothing real.
            temperature=DECODE_TEMPERATURES,
            max_new_tokens=MAX_NEW_TOKENS,
            condition_on_previous_text=False,
        )
        return _scored(segments, info, self.language)

    def transcribe_detailed(self, pcm16: np.ndarray) -> tuple[str, str]:
        r = self.transcribe_scored(pcm16)
        return r.text, r.language

    def transcribe(self, pcm16: np.ndarray) -> str:
        return self.transcribe_detailed(pcm16)[0]

    async def atranscribe(self, pcm16: np.ndarray) -> str:
        return await asyncio.to_thread(self.transcribe, pcm16)

    async def atranscribe_detailed(self, pcm16: np.ndarray) -> tuple[str, str]:
        return await asyncio.to_thread(self.transcribe_detailed, pcm16)

    async def atranscribe_scored(self, pcm16: np.ndarray) -> Transcript:
        return await asyncio.to_thread(self.transcribe_scored, pcm16)



# When a transcript is noise whisper put words to rather than speech. Tuned
# with scripts/eval_noise_transcripts.py (small / small.en, Kokoro speech,
# synthetic noise): every noise clip whisper put words to scored
# no_speech >= 0.79; real speech (English and Hindi answers, redirects and
# requests, clean and under 10 dB of noise) <= 0.56, logprob >= -1.05 and
# compression <= 1.0. A wrong language alone proves nothing: short Hindi
# answers come back as Japanese or Russian (no_speech up to 0.48) and
# "Yeah." as Spanish. In the Hindi mode, English that whisper mangles into
# Devanagari scores logprob -1.49 to -1.60 and counts as noise: it was
# never usable text.
NOISE_NO_SPEECH = 0.7          # whisper is sure there was no speech
NOISE_LOGPROB = -1.4           # every decode, fallbacks included, was a guess
NOISE_COMPRESSION = 2.4        # a repetition loop (whisper's own threshold)
NOISE_FOREIGN_LANG_P = 0.5     # unsure of the language...
NOISE_FOREIGN_NO_SPEECH = 0.5  # ...and of there being speech at all


def noise_reason(r: Transcript, settings, expected_langs) -> str | None:
    """Why `r` is noise rather than speech, or None. Only scored, non-empty
    transcripts are judged; an unscored one (a double, an old transcriber)
    is taken at its word. `expected_langs`: what the user speaks in the
    current language mode — a language whisper merely guessed at, outside
    those, counts against the transcript."""
    if not getattr(settings, "noise_transcript_filter", True) or not (r.text or "").strip():
        return None
    if r.no_speech_prob is None:
        return None
    if r.no_speech_prob >= NOISE_NO_SPEECH:
        return "no_speech"
    if r.avg_logprob <= NOISE_LOGPROB:
        return "logprob"
    if r.compression_ratio >= NOISE_COMPRESSION:
        return "repetition"
    if (r.language not in expected_langs and r.language_probability is not None
            and r.language_probability < NOISE_FOREIGN_LANG_P and r.no_speech_prob >= NOISE_FOREIGN_NO_SPEECH):
        return "language"
    return None
