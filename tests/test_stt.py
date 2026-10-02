import numpy as np
import pytest
import soundfile as sf

from veronica.speech.stt import Transcriber, Transcript


class FakeSeg:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self, name, device, compute_type):
        self.name = name

    def transcribe(self, audio, beam_size, language, vad_filter, **kw):
        return iter([FakeSeg(" hello "), FakeSeg("world")]), None


def test_transcribe_joins_segments(monkeypatch):
    monkeypatch.setattr(Transcriber, "_model_cls", FakeModel)
    t = Transcriber("base.en")
    assert t.transcribe(np.zeros(16000, dtype=np.int16)) == "hello world"


async def test_atranscribe(monkeypatch):
    monkeypatch.setattr(Transcriber, "_model_cls", FakeModel)
    t = Transcriber("base.en")
    assert await t.atranscribe(np.zeros(16000, dtype=np.int16)) == "hello world"


def test_transcribe_detailed_reports_language(monkeypatch):
    calls = []

    class Info:
        language = "hi"

    class Seg:
        def __init__(self, t):
            self.text = t

    class M:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kw):
            calls.append(kw)
            return iter([Seg(" नमस्ते ")]), Info()

    monkeypatch.setattr(Transcriber, "_model_cls", M)
    t = Transcriber("small", language=None)
    assert t.transcribe_detailed(np.zeros(16000, dtype=np.int16)) == ("नमस्ते", "hi")
    assert calls[-1]["language"] is None
    t.set_language("hi")
    t.transcribe(np.zeros(16000, dtype=np.int16))
    assert calls[-1]["language"] == "hi"
    t2 = Transcriber("small.en")
    t2.transcribe(np.zeros(16000, dtype=np.int16))
    assert calls[-1]["language"] == "en"


async def test_atranscribe_detailed(monkeypatch):
    monkeypatch.setattr(Transcriber, "_model_cls", FakeModel)
    t = Transcriber("base.en")
    assert await t.atranscribe_detailed(np.zeros(16000, dtype=np.int16)) == ("hello world", "en")


@pytest.mark.live
def test_real_whisper_on_fixture(monkeypatch):
    from faster_whisper import WhisperModel
    monkeypatch.setattr(Transcriber, "_model_cls", WhisperModel)
    pcm, sr = sf.read("tests/fixtures/speech_1s.wav", dtype="int16")
    text = Transcriber("base.en").transcribe(pcm).lower()
    assert "time" in text


# -- confidence: what whisper says about its own transcript ---------------------

class ScoredSeg:
    def __init__(self, text, no_speech_prob, avg_logprob, compression_ratio, tokens=5):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio
        self.tokens = list(range(tokens))


def _scored_model(segs, language="de", language_probability=0.41):
    class Info:
        pass

    info = Info()
    info.language = language
    info.language_probability = language_probability

    class M:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kw):
            return iter(segs), info

    return M


def test_transcribe_scored_reports_whisper_confidence(monkeypatch):
    monkeypatch.setattr(Transcriber, "_model_cls", _scored_model([
        ScoredSeg(" Ich küsse, ", 0.2, -0.5, 1.5, tokens=6),
        ScoredSeg("küsse, küsse. ", 0.7, -1.1, 2.6, tokens=2),
    ]))
    r = Transcriber("small", language=None).transcribe_scored(np.zeros(16000, dtype=np.int16))
    assert isinstance(r, Transcript)
    assert r.text == "Ich küsse, küsse, küsse."
    assert r.language == "de" and r.language_probability == pytest.approx(0.41)
    assert r.no_speech_prob == pytest.approx(0.7)          # the worst window
    assert r.compression_ratio == pytest.approx(2.6)       # the worst segment
    assert r.avg_logprob == pytest.approx((-0.5 * 6 - 1.1 * 2) / 8)   # token-weighted


def test_transcribe_scored_pinned_language_and_empty(monkeypatch):
    monkeypatch.setattr(Transcriber, "_model_cls", _scored_model([], language="en", language_probability=1.0))
    r = Transcriber("small.en").transcribe_scored(np.zeros(16000, dtype=np.int16))
    assert r.text == "" and r.language == "en"
    assert r.no_speech_prob is None and r.avg_logprob is None and r.compression_ratio is None


def test_transcribe_scored_without_scores_is_unscored(monkeypatch):
    # FakeModel's segments carry text only (and info is None): no scores, no
    # judgement — never mistaken for noise.
    monkeypatch.setattr(Transcriber, "_model_cls", FakeModel)
    r = Transcriber("base.en").transcribe_scored(np.zeros(16000, dtype=np.int16))
    assert r == Transcript("hello world", "en")


async def test_atranscribe_scored_and_detailed_agree(monkeypatch):
    monkeypatch.setattr(Transcriber, "_model_cls", _scored_model([ScoredSeg(" yes ", 0.1, -0.3, 0.8)], "en", 0.9))
    t = Transcriber("small", language=None)
    r = await t.atranscribe_scored(np.zeros(16000, dtype=np.int16))
    assert (r.text, r.language) == ("yes", "en")
    assert t.transcribe_detailed(np.zeros(16000, dtype=np.int16)) == ("yes", "en")


def test_decoding_is_bounded_so_noise_cannot_stall_a_turn():
    """Whisper's default temperature fallback re-ran a hallucination loop on
    3 s of pink noise for 297 s — long enough to stall a confirm and the task
    behind it. Every transcription is bounded: one retry and a token cap."""
    from veronica.speech import stt

    seen = {}

    class Model:
        def transcribe(self, audio, **kw):
            seen.update(kw)
            return iter([]), None

    t = stt.Transcriber.__new__(stt.Transcriber)
    t._model, t.language = Model(), "en"
    t.transcribe_scored(np.zeros(16000, dtype=np.int16))
    assert seen["temperature"] == stt.DECODE_TEMPERATURES and len(stt.DECODE_TEMPERATURES) <= 2
    assert seen["max_new_tokens"] == stt.MAX_NEW_TOKENS
    assert seen["condition_on_previous_text"] is False
