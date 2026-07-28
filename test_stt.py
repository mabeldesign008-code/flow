"""Tests for the pluggable STT layer."""

import asyncio
import io
import wave

import numpy as np
import pytest

from stt.base import TranscriptionResult, WordInfo, float_to_wav_bytes
from stt.assemblyai_client import AssemblyAIClient, _clean_keyterms
from stt.local_client import LocalSherpaClient, _ensure_16k
from stt.router import EnginePolicy, EngineRouter
from stt.dictionary import UserDictionary


# ── base ──────────────────────────────────────────────────────────────────

class TestWavEncoding:
    def test_roundtrip(self):
        sr = 16000
        tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        data = float_to_wav_bytes(tone, sr)

        with wave.open(io.BytesIO(data), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == sr
            assert wf.getnframes() == sr

    def test_clips_instead_of_wrapping(self):
        """np.int16(x * 32767) wraps on overflow and makes loud noise bursts."""
        loud = np.array([3.0, -3.0, 0.0], dtype=np.float32)
        data = float_to_wav_bytes(loud, 16000)
        with wave.open(io.BytesIO(data), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(3), dtype=np.int16)
        assert pcm[0] == 32767
        assert pcm[1] == -32767
        assert pcm[2] == 0

    def test_empty(self):
        assert len(float_to_wav_bytes(np.array([], dtype=np.float32), 16000)) > 0


class TestResult:
    def test_failure_is_not_ok(self):
        r = TranscriptionResult.failure("boom", "eng", "m")
        assert not r.ok and r.error == "boom" and r.is_empty

    def test_empty_ok_is_distinct_from_failure(self):
        """Genuine silence must be distinguishable from an error."""
        silent = TranscriptionResult(text="", ok=True)
        failed = TranscriptionResult.failure("network down")
        assert silent.is_empty and silent.ok
        assert failed.is_empty and not failed.ok

    def test_low_confidence_words(self):
        r = TranscriptionResult(
            text="deploy to cuban artists",
            words=[
                WordInfo("deploy", 0.99), WordInfo("to", 0.98),
                WordInfo("cuban", 0.31), WordInfo("artists", 0.29),
            ],
        )
        assert [w.text for w in r.low_confidence_words()] == ["cuban", "artists"]


# ── AssemblyAI ────────────────────────────────────────────────────────────

class TestKeyterms:
    def test_drops_long_phrases(self):
        assert _clean_keyterms(["ok", "one two three four five six seven"]) == ["ok"]

    def test_dedupes_case_insensitively(self):
        assert _clean_keyterms(["Groq", "groq", "GROQ"]) == ["Groq"]

    def test_respects_limit(self):
        assert len(_clean_keyterms([f"t{i}" for i in range(50)], limit=10)) == 10


class TestAssemblyAIClient:
    def test_unconfigured_fails_cleanly(self):
        c = AssemblyAIClient(api_key="")
        r = asyncio.run(c.transcribe(np.zeros(1600, dtype=np.float32), 16000))
        assert not r.ok and "key" in r.error.lower()

    def test_empty_audio_fails_cleanly(self):
        c = AssemblyAIClient(api_key="k")
        r = asyncio.run(c.transcribe(np.array([], dtype=np.float32), 16000))
        assert not r.ok

    def test_parse_extracts_words_and_confidence(self):
        c = AssemblyAIClient(api_key="k")
        r = c._parse(
            {
                "text": "Hello world",
                "confidence": 0.94,
                "language_code": "en",
                "words": [
                    {"text": "Hello", "confidence": 0.99, "start": 0, "end": 400},
                    {"text": "world", "confidence": 0.89, "start": 410, "end": 800},
                ],
            },
            started=0.0, duration=1.0,
        )
        assert r.ok and r.text == "Hello world"
        assert len(r.words) == 2 and r.words[1].confidence == 0.89

    def test_parse_empty_is_ok_not_error(self):
        c = AssemblyAIClient(api_key="k")
        r = c._parse({"text": "", "words": []}, started=0.0, duration=1.0)
        assert r.ok and r.is_empty

    def test_keyterms_only_for_supported_models(self):
        assert AssemblyAIClient(model="universal-3-5-pro").get_info()["keyterms_supported"]
        assert not AssemblyAIClient(model="universal-2").get_info()["keyterms_supported"]

    def test_set_api_key_resets_client(self):
        c = AssemblyAIClient(api_key="old")
        c._client = object()
        c.set_api_key("new")
        assert c.api_key == "new" and c._client is None


# ── local ─────────────────────────────────────────────────────────────────

class TestLocalClient:
    def test_strips_sensevoice_tags(self):
        c = LocalSherpaClient(model_type="sensevoice")
        assert c._clean("<|en|><|NEUTRAL|><|Speech|><|woitn|>hello world") == "hello world"

    def test_keeps_currency_and_punctuation(self):
        """The old _enforce_english() deleted these entirely."""
        c = LocalSherpaClient(model_type="sensevoice")
        for s in ["It costs \u20ac20 \u2014 that\u2019s fine, caf\u00e9",
                  "\u2014 \u20ac50", "caf\u00e9"]:
            assert c._clean(s) == s

    def test_does_not_eat_bracketed_text(self):
        """The old catch-all [A-Z_]+ pattern ate things like <|A|>."""
        c = LocalSherpaClient(model_type="sensevoice")
        assert "A" in c._clean("the value is <|A|> ok")

    def test_missing_model_reports_error_not_hang(self):
        c = LocalSherpaClient(model_type="parakeet", model_dir="/nonexistent", load_timeout=2)
        c._load_model()
        r = asyncio.run(c.transcribe(np.zeros(1600, dtype=np.float32), 16000))
        assert not r.ok and r.error and "load" in r.error.lower()

    def test_resample_to_16k(self):
        audio = np.random.randn(48000).astype(np.float32)
        out, sr = _ensure_16k(audio, 48000)
        assert sr == 16000 and abs(len(out) - 16000) < 100

    def test_resample_noop_at_16k(self):
        audio = np.zeros(1600, dtype=np.float32)
        out, sr = _ensure_16k(audio, 16000)
        assert sr == 16000 and len(out) == 1600


# ── router ────────────────────────────────────────────────────────────────

class _FakeEngine:
    def __init__(self, result, name="fake"):
        self.result, self.name, self.calls = result, name, 0

    async def transcribe(self, samples, sample_rate, keyterms=None):
        self.calls += 1
        return self.result

    async def close(self): pass
    def preload(self): pass
    def get_info(self): return {"engine": self.name}


class TestRouter:
    def _audio(self):
        return np.zeros(1600, dtype=np.float32), 16000

    def test_cloud_success_skips_local(self):
        cloud = _FakeEngine(TranscriptionResult(text="from cloud", ok=True))
        local = _FakeEngine(TranscriptionResult(text="from local", ok=True))
        cloud.is_configured = True
        r = asyncio.run(EngineRouter(cloud, local).transcribe(*self._audio()))
        assert r.text == "from cloud" and local.calls == 0

    def test_falls_back_on_cloud_failure(self):
        cloud = _FakeEngine(TranscriptionResult.failure("503"))
        cloud.is_configured = True
        local = _FakeEngine(TranscriptionResult(text="rescued", ok=True))
        r = asyncio.run(EngineRouter(cloud, local).transcribe(*self._audio()))
        assert r.ok and r.text == "rescued" and local.calls == 1

    def test_empty_but_ok_does_not_trigger_fallback(self):
        """Genuine silence must not cause a wasteful local retry."""
        cloud = _FakeEngine(TranscriptionResult(text="", ok=True))
        cloud.is_configured = True
        local = _FakeEngine(TranscriptionResult(text="hallucinated", ok=True))
        r = asyncio.run(EngineRouter(cloud, local).transcribe(*self._audio()))
        assert r.is_empty and r.ok and local.calls == 0

    def test_local_only_never_calls_cloud(self):
        cloud = _FakeEngine(TranscriptionResult(text="cloud", ok=True))
        cloud.is_configured = True
        local = _FakeEngine(TranscriptionResult(text="local", ok=True))
        router = EngineRouter(cloud, local, policy=EnginePolicy.LOCAL_ONLY)
        r = asyncio.run(router.transcribe(*self._audio()))
        assert r.text == "local" and cloud.calls == 0

    def test_cloud_only_does_not_fall_back(self):
        cloud = _FakeEngine(TranscriptionResult.failure("bad key"))
        cloud.is_configured = True
        local = _FakeEngine(TranscriptionResult(text="local", ok=True))
        router = EngineRouter(cloud, local, policy=EnginePolicy.CLOUD_ONLY)
        r = asyncio.run(router.transcribe(*self._audio()))
        assert not r.ok and local.calls == 0

    def test_both_fail_reports_cloud_cause(self):
        cloud = _FakeEngine(TranscriptionResult.failure("insufficient credit"))
        cloud.is_configured = True
        local = _FakeEngine(TranscriptionResult.failure("no model"))
        r = asyncio.run(EngineRouter(cloud, local).transcribe(*self._audio()))
        assert not r.ok and "insufficient credit" in r.error

    def test_no_key_uses_local(self):
        cloud = _FakeEngine(TranscriptionResult(text="cloud", ok=True))
        cloud.is_configured = False
        local = _FakeEngine(TranscriptionResult(text="local", ok=True))
        r = asyncio.run(EngineRouter(cloud, local).transcribe(*self._audio()))
        assert r.text == "local"

    def test_failure_counter(self):
        cloud = _FakeEngine(TranscriptionResult.failure("x"))
        cloud.is_configured = True
        router = EngineRouter(cloud, None, policy=EnginePolicy.CLOUD_ONLY)
        for _ in range(3):
            asyncio.run(router.transcribe(*self._audio()))
        assert router.consecutive_cloud_failures == 3


# ── dictionary ────────────────────────────────────────────────────────────

class TestDictionary:
    def test_add_and_persist(self, tmp_path):
        p = tmp_path / "d.txt"
        d = UserDictionary(p)
        assert d.add("Kubernetes") and d.add("WhisprFlow")
        assert UserDictionary(p).terms == ["Kubernetes", "WhisprFlow"]

    def test_rejects_duplicates_case_insensitively(self, tmp_path):
        d = UserDictionary(tmp_path / "d.txt")
        assert d.add("Groq") and not d.add("groq")
        assert len(d) == 1

    def test_rejects_overlong_phrases(self, tmp_path):
        d = UserDictionary(tmp_path / "d.txt")
        assert not d.add("one two three four five six seven")
        assert not d.add("x" * 60)

    def test_comments_ignored_on_load(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("# comment\nAlpha\n\nBeta\n", encoding="utf-8")
        assert UserDictionary(p).terms == ["Alpha", "Beta"]

    def test_remove(self, tmp_path):
        d = UserDictionary(tmp_path / "d.txt")
        d.add("Temp")
        assert d.remove("temp") and len(d) == 0

    def test_hotwords_file_is_uppercase(self, tmp_path):
        d = UserDictionary(tmp_path / "d.txt")
        d.add("Kubernetes")
        path = d.write_hotwords_file(tmp_path / "hw.txt")
        assert "KUBERNETES" in open(path).read()

    def test_prompt_line(self, tmp_path):
        d = UserDictionary(tmp_path / "d.txt")
        d.add_many(["Groq", "Parakeet"])
        assert d.as_prompt_line() == "Groq, Parakeet"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
