"""Room noise that whisper put words to ("Ich küsse, küsse, küsse.") is
silence: never a confirm answer — so never an approval and never a redirect
that ends the running task — never a request, never a follow-up, never a
dictated line. A real answer or redirect still gets through, and an `other`
answer that isn't plausibly an instruction is asked about once more instead
of being run."""
import logging

import numpy as np
import pytest

from tests.test_orchestrator import STT, TTS, Brain, Player, Rec, Wake
from veronica.config import Settings
from veronica.orchestrator import Orchestrator
from veronica.speech.stt import Transcript, expected_langs, noise_reason
from veronica.tools import mac as mac_tools_mod

PCM = np.zeros(1, np.int16)


def speech(text: str, lang: str = "en", lang_p: float = 0.95) -> Transcript:
    return Transcript(text, lang, lang_p, no_speech_prob=0.02, avg_logprob=-0.25, compression_ratio=0.9)


def noise(text: str = "Ich küsse, küsse, küsse.") -> Transcript:
    # The scores whisper gave the real thing, near enough.
    return Transcript(text, "de", 0.41, no_speech_prob=0.82, avg_logprob=-1.3, compression_ratio=2.6)


class ScoredSTT(STT):
    """Hands out Transcripts (with whisper's scores) in order."""

    def __init__(self, results):
        super().__init__([])
        self.results = list(results)

    async def atranscribe_scored(self, pcm):
        return self.results.pop(0) if self.results else Transcript("", "en")

    async def atranscribe(self, pcm):
        return (await self.atranscribe_scored(pcm)).text


def build(results, captures=None, mode="auto", **settings):
    events = []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0, **settings),
        wake=Wake(), recorder=Rec([PCM] * (len(results) if captures is None else captures)),
        stt=ScoredSTT(results), brain=Brain(), tts=TTS(), player=Player(), on_state=lambda s: None,
        on_event=lambda k, p: events.append((k, p)), language=mode,
    )
    return o, events


# -- the judgement -------------------------------------------------------------

def test_the_reported_noise_is_noise():
    why = noise_reason(noise(), Settings(), ("en", "hi"))
    assert why


@pytest.mark.parametrize("r", [
    speech("Yes."), speech("No."), speech("Haan.", "hi"), speech("हाँ।", "hi", 0.6),
    speech("No, use Safari instead."),
    # Measured (scripts/eval_noise_transcripts.py): the worst real answers.
    # "नहीं, रहने दो" under clatter, "Nope." under clatter, and "Yeah." heard
    # as Spanish.
    Transcript("नहीं रहने दो", "hi", 0.92, no_speech_prob=0.56, avg_logprob=-0.33, compression_ratio=0.86),
    Transcript("NOPE", "en", 1.0, no_speech_prob=0.05, avg_logprob=-1.05, compression_ratio=0.33),
    Transcript("Ya.", "es", 0.37, no_speech_prob=0.20, avg_logprob=-0.89, compression_ratio=0.27),
    Transcript("はじ", "ja", 0.2, no_speech_prob=0.48, avg_logprob=-0.8, compression_ratio=0.3),   # "हाँ जी"
])
def test_real_speech_is_not_noise(r):
    assert noise_reason(r, Settings(), ("en", "hi")) is None


@pytest.mark.parametrize("r, why", [
    (Transcript("you", "en", 1.0, no_speech_prob=0.9, avg_logprob=-0.8, compression_ratio=0.6), "no_speech"),
    (Transcript("The end.", "en", 1.0, no_speech_prob=0.1, avg_logprob=-1.6, compression_ratio=0.9), "logprob"),
    (Transcript("la la la la la la la la", "en", 1.0, no_speech_prob=0.1, avg_logprob=-0.3,
                compression_ratio=2.9), "repetition"),
    (Transcript("Merci.", "fr", 0.4, no_speech_prob=0.55, avg_logprob=-0.3, compression_ratio=0.6), "language"),
])
def test_each_signal_rejects_on_its_own(r, why):
    assert why in noise_reason(r, Settings(), ("en", "hi"))


def test_language_only_counts_when_whisper_was_unsure():
    # Confidently another language is a person speaking it — the speaker
    # check's call, not a noise call.
    sure = Transcript("Bonjour, ça va?", "fr", 0.97, no_speech_prob=0.05, avg_logprob=-0.3, compression_ratio=0.9)
    assert noise_reason(sure, Settings(), ("en", "hi")) is None
    # ...and a language the mode expects is never held against it.
    unsure_hi = Transcript("हाँ।", "hi", 0.3, no_speech_prob=0.1, avg_logprob=-0.4, compression_ratio=0.6)
    assert noise_reason(unsure_hi, Settings(), ("en", "hi")) is None


def test_unscored_and_empty_transcripts_are_left_alone():
    assert noise_reason(Transcript("Ich küsse, küsse, küsse.", "en"), Settings(), ("en",)) is None
    assert noise_reason(Transcript("", "de", 0.1, 0.99, -3.0, 4.0), Settings(), ("en",)) is None


def test_expected_languages_follow_the_mode():
    assert expected_langs("en") == ("en",)
    # whisper labels spoken Hindi as Urdu often enough that it counts
    assert set(expected_langs("auto")) == set(expected_langs("hi")) == {"en", "hi", "ur"}


def test_the_filter_can_be_turned_off():
    assert noise_reason(noise(), Settings(noise_transcript_filter=False), ("en", "hi")) is None


# -- confirm answers ---------------------------------------------------------------

async def test_confirm_noise_is_no_answer_and_listens_again(caplog):
    caplog.set_level(logging.INFO, logger="veronica.orchestrator")
    o, _ = build([noise(), speech("Yes.")])
    r = await o.confirm("Bash: ls")
    assert r.outcome == "approved"
    assert o.tts.said == ["Run Bash: ls?"]
    assert "stt: treated as noise" in caplog.text and "'Ich küsse" in caplog.text
    assert "no_speech=0.82" in caplog.text and "lang=de/0.41" in caplog.text


async def test_confirm_noise_twice_is_the_usual_skip_never_a_redirect():
    o, events = build([noise(), noise("Thank you. Thank you. Thank you.")])
    r = await o.confirm("Bash: ls")
    assert r.outcome == "denied" and not r and r.heard == ""
    assert o.tts.said == ["Run Bash: ls?", "Okay, skipping that."]
    assert [p["decision"] for k, p in events if k == "tool" and "decision" in p][-1] == "declined"


async def test_confirm_noise_that_says_yes_is_still_no_approval():
    """Never weaken the gate: a "yes" whisper made out of noise is silence."""
    o, _ = build([noise("Yes."), noise("Yes, yes, yes, yes, yes.")])
    r = await o.confirm("Bash: rm -rf build")
    assert r.outcome == "denied" and not r


async def test_confirm_a_real_redirect_still_redirects():
    o, _ = build([speech("No, open it in Safari instead.")])
    r = await o.confirm("Open Chrome")
    assert r.outcome == "other" and r.heard == "No, open it in Safari instead."


@pytest.mark.parametrize("heard", ["Safari.", "Um, the window.", "küsse küsse küsse"])
async def test_confirm_an_unclear_other_is_asked_once_more(heard, caplog):
    caplog.set_level(logging.INFO, logger="veronica.orchestrator")
    o, _ = build([speech(heard), speech("Yes.")])
    r = await o.confirm("Open Chrome")
    assert r.outcome == "approved"
    assert o.tts.said == ["Run Open Chrome?", "Sorry, yes or no?"]
    assert "unclear" in caplog.text


async def test_confirm_unclear_twice_is_skipped_not_run():
    o, _ = build([speech("Safari."), speech("Mm, Chrome.")])
    r = await o.confirm("Open Chrome")
    assert r.outcome == "denied" and not r
    assert o.tts.said == ["Run Open Chrome?", "Sorry, yes or no?", "Okay, skipping that."]


async def test_confirm_the_reask_can_carry_a_real_redirect():
    o, _ = build([speech("Safari."), speech("No, use Safari instead.")])
    r = await o.confirm("Open Chrome")
    assert r.outcome == "other" and r.heard == "No, use Safari instead."


async def test_confirm_a_redirect_in_a_language_she_doesnt_expect_is_unclear():
    o, _ = build([speech("Öffne es bitte in Safari.", "de", 0.9), speech("No.")])
    r = await o.confirm("Open Chrome")
    assert r.outcome == "denied"
    assert "Sorry, yes or no?" in o.tts.said


async def test_confirm_with_the_filter_off_takes_text_at_its_word():
    o, _ = build([noise()], noise_transcript_filter=False)
    r = await o.confirm("Open Chrome")
    assert r.outcome == "other"


@pytest.mark.parametrize("heard", ["No, open it in Safari instead.", "what will that do?",
                                   "haan lekin Chrome mein", "Yes, but in Chrome."])
def test_plausible_redirects(heard):
    assert Orchestrator.redirect_words(heard) >= 2


@pytest.mark.parametrize("heard", ["Safari.", "um, the window", "küsse küsse küsse", "no no no thanks"])
def test_implausible_redirects(heard):
    assert Orchestrator.redirect_words(heard) < 2


# -- requests, follow-ups, dictation -----------------------------------------------

async def test_a_noise_request_runs_no_brain_turn(caplog):
    caplog.set_level(logging.INFO, logger="veronica.orchestrator")
    o, events = build([noise()], captures=1)
    await o.one_turn()
    assert o.brain.asked == []
    assert ("heard", "Ich küsse, küsse, küsse.") not in events
    assert "stt: treated as noise" in caplog.text


async def test_a_noise_follow_up_ends_the_turn_quietly():
    o, _ = build([speech("Tell me a joke."), noise()], captures=2)
    await o.one_turn()
    assert o.brain.asked == ["Tell me a joke."]
    assert "Sorry, didn't catch that." not in o.tts.said


async def test_noise_is_not_dictated(monkeypatch):
    typed = []
    monkeypatch.setattr(mac_tools_mod, "dictate_type",
                        lambda t: typed.append(t) or {"content": [{"type": "text", "text": "ok"}]})
    o, _ = build([speech("dictate"), noise(), speech("hello there"), speech("stop dictation")], captures=4)
    await o.one_turn()
    assert typed == ["hello there"]
