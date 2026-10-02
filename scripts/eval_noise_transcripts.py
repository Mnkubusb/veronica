"""Measure the noise-transcript filter on synthetic audio.

    uv run python scripts/eval_noise_transcripts.py [--out DIR] [--quick]

Room noise that gets past the VAD is still handed to whisper, which then
"hears" something — "Ich küsse, küsse, küsse." — and that text used to
become a confirm answer (a redirect) or a request. This runs real
faster-whisper, in each language mode Veronica uses, over synthetic noise
(pink, keyboard clatter, music, babble, a TV-like bed, plus the voice-
isolation fixtures) and over Kokoro-synthesised English and Hindi speech —
short answers ("yes", "no", "haan"), redirects ("no, use Safari instead"),
requests — clean and under noise, and reports what stt.noise_reason makes
of each: noise rejected %, real speech kept %, and every transcript it got
wrong with its scores. No microphone is touched. Clips are cached in DIR
(default ~/.veronica/eval); raw scores land in DIR/noise_transcripts_<mode>.json
(--report-only re-scores those). Slow: whisper's temperature fallback can spend
minutes looping on a noise clip, so --quick is a sensible first run.
Babble and the TV bed are real (other) voices: whisper hears words in them
with confidence, so they are reported but are the speaker check's job, not
this filter's."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import eval_voice_isolation as evi  # (Clips, at, i16, rng)

from veronica.config import Settings, settings
from veronica.speech.stt import Transcriber, expected_langs, noise_reason, stt_spec

EN_SHORT = ["Yes.", "No.", "Yeah.", "Yep.", "Nope.", "Okay.", "Sure.", "Go ahead.", "Yes, do it.",
            "No thanks.", "Do it.", "Not now.", "Cancel.", "Stop."]
EN_REDIRECT = ["No, use Safari instead.", "No, open it in Safari instead.", "Yes, but in Chrome.",
               "What will that do?", "No, send it to Rahul instead.", "Wait, use the other window."]
EN_REQUEST = ["What time is it in London right now", "Set a timer for ten minutes",
              "Open Netflix and play Breaking Bad", "Read my latest email"]
HI_SHORT = ["हाँ।", "नहीं।", "हाँ, कर दो।", "नहीं, रहने दो।", "ठीक है।", "हाँ जी।", "मत करो।"]
HI_REDIRECT = ["नहीं, सफारी में खोलो।", "नहीं, राहुल को भेजो।"]
HI_REQUEST = ["आज मौसम कैसा है", "दस मिनट का टाइमर लगाओ"]
EN_VOICES = ["af_heart", "am_adam", "bf_emma"]
HI_VOICES = ["hf_beta", "hf_alpha", "hm_omega"]
NOISES = ["pink", "clatter", "music", "babble", "tv"]


class HindiClips(evi.Clips):
    def say_hi(self, text: str, voice: str) -> np.ndarray:
        p = self.out / "clips" / f"{voice}_hi_{abs(hash(text)) % 10**8}.wav"
        if p.exists():
            return sf.read(p, dtype="float32")[0]
        from scipy.signal import resample_poly

        s, _sr = self.k.create(text, voice=voice, lang="hi")
        x = resample_poly(np.asarray(s, np.float32), 2, 3)
        x = (x / np.max(np.abs(x)) * 0.5).astype(np.float32)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, evi.i16(x), 16000)
        return x


def pad(x: np.ndarray) -> np.ndarray:
    """Speech as the recorder hands it over: a little lead-in and tail."""
    z = np.zeros(int(0.3 * 16000), np.float32)
    return np.concatenate([z, x, z])


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    off = int(evi.rng.integers(0, noise.size - speech.size - 1))
    n = noise[off:off + speech.size]
    n = n / (evi.rmsv(n) + 1e-12) * evi.rmsv(speech) / (10 ** (snr_db / 20))
    return (speech + n).astype(np.float32)


def clips(c: HindiClips, quick: bool) -> list[dict]:
    out = []
    noises = {n: c.noise(n) for n in NOISES}
    for n, x in noises.items():
        for level in (0.003, 0.01, 0.03):
            for secs in ((1.5, 3.0) if quick else (1.5, 3.0, 6.0)):
                for _ in range(1 if quick else 2):
                    off = int(evi.rng.integers(0, x.size - int(secs * 16000)))
                    out.append({"kind": "noise", "src": n, "label": f"{n}@{level} {secs}s",
                                "pcm": evi.i16(evi.at(x[off:off + int(secs * 16000)], level))})
    fx = Path(__file__).parent.parent / "tests" / "fixtures" / "voice"
    for f in ("noise_pink.flac", "noise_babble.flac"):
        out.append({"kind": "noise", "src": f.split(".")[0], "label": f, "pcm": sf.read(fx / f, dtype="int16")[0]})
    speech = []
    for group, lines, voices, say in (
        ("en_short", EN_SHORT, EN_VOICES, c.say), ("en_redirect", EN_REDIRECT, EN_VOICES, c.say),
        ("en_request", EN_REQUEST, EN_VOICES[:1], c.say),
        ("hi_short", HI_SHORT, HI_VOICES, c.say_hi), ("hi_redirect", HI_REDIRECT, HI_VOICES, c.say_hi),
        ("hi_request", HI_REQUEST, HI_VOICES[:1], c.say_hi),
    ):
        for v in voices[:1] if quick else voices:
            for line in lines:
                speech.append((group, line, v, pad(say(line, v))))
    for group, line, v, x in speech:
        lang = "hi" if group.startswith("hi") else "en"
        out.append({"kind": "speech", "src": group, "lang": lang, "label": f"{v}: {line}",
                    "pcm": evi.i16(evi.at(x, 0.05))})
        if group.endswith(("short", "redirect")):
            for n in ("clatter",) if quick else ("pink", "clatter", "music"):
                out.append({"kind": "speech", "src": group + "+noise", "lang": lang,
                            "label": f"{v}: {line} +{n} 10dB",
                            "pcm": evi.i16(evi.at(mix(x, noises[n], 10.0), 0.05))})
    out.append({"kind": "speech", "src": "en_short", "lang": "en", "label": "fixture me_yes",
                "pcm": sf.read(fx / "me_yes.flac", dtype="int16")[0]})
    return out


def run(out: Path, quick: bool, modes) -> dict:
    import time

    c = HindiClips(out)
    items = clips(c, quick)
    print(f"{len(items)} clips", flush=True)
    results = {}
    for mode in modes:
        t0 = time.monotonic()
        model, lang, _partial = stt_spec(settings, mode)
        t = Transcriber(model, lang)
        rows = []
        for it in items:
            if mode == "en" and it.get("lang") == "hi":
                continue   # English mode doesn't hear Hindi by design
            r = t.transcribe_scored(it["pcm"])
            rows.append({k: v for k, v in it.items() if k != "pcm"} | {"r": r._asdict()})
            if len(rows) % 10 == 0:
                print(f"[{mode}] {len(rows)} ({time.monotonic() - t0:.0f} s)", flush=True)
        results[mode] = rows
        (out / f"noise_transcripts_{mode}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"[{mode}] {len(rows)} clips transcribed", flush=True)
    return results


def report(results: dict, s: Settings) -> None:
    from veronica.speech.stt import Transcript

    for mode, rows in results.items():
        expected = expected_langs(mode)
        noise_n = noise_rej = speech_n = speech_kept = 0
        heard_noise = 0
        wrong = []
        per_src: dict[str, list[int]] = {}
        for row in rows:
            r = Transcript(**row["r"])
            why = noise_reason(r, s, expected)
            acc = per_src.setdefault(f"{row['kind']}:{row['src']}", [0, 0, 0])
            acc[0] += 1
            if row["kind"] == "noise":
                if not r.text.strip():
                    acc[2] += 1   # whisper itself heard nothing: already silence
                    continue
                heard_noise += 1
                rejected = bool(why)
                acc[1] += rejected
                if row["src"] not in ("babble", "tv", "noise_babble"):
                    noise_n += 1
                    noise_rej += rejected
                if not rejected:
                    wrong.append(("noise kept", row["label"], r))
            else:
                speech_n += 1
                speech_kept += not why
                acc[1] += not why
                if why:
                    wrong.append((f"speech rejected ({why})", row["label"], r))
        print(f"\n== mode {mode} (expected {expected}) ==")
        print(f"non-voice noise that whisper put words to: rejected {noise_rej}/{noise_n}"
              f" ({100 * noise_rej / max(1, noise_n):.0f}%)")
        print(f"real speech: kept {speech_kept}/{speech_n} ({100 * speech_kept / max(1, speech_n):.0f}%)")
        for k, (n, good, empty) in sorted(per_src.items()):
            what = "rejected" if k.startswith("noise") else "kept"
            print(f"  {k:28s} n={n:3d} {what}={good:3d} empty={empty}")
        for tag, label, r in wrong:
            print(f"  {tag}: {label} -> {r.text!r} ns={_f(r.no_speech_prob)} lp={_f(r.avg_logprob)} "
                  f"cr={_f(r.compression_ratio)} lang={r.language}/{_f(r.language_probability)}")


def _f(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / ".veronica" / "eval")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="re-score the saved transcripts")
    ap.add_argument("--modes", default="en,auto,hi", help="language modes to run (comma-separated)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    modes = a.modes.split(",")
    if a.report_only:
        results = {m: json.loads(p.read_text()) for m in modes if (p := a.out / f"noise_transcripts_{m}.json").exists()}
    else:
        results = run(a.out, a.quick, modes)
    report(results, Settings())


if __name__ == "__main__":
    main()
