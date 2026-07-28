"""
AssemblyAI transcription engine.

Model default is `universal-3-5-pro` -- AssemblyAI's most accurate async
model (~7.7% normalized WER on hard code-switched audio, ~2-5% on clean
English). Compare SenseVoice-Small at ~14.7% English WER, which is what
this replaces.

Two-step API: upload raw bytes to /v2/upload, then POST /v2/transcript and
poll. We deliberately do NOT use the official SDK -- it is synchronous and
would block the asyncio loop. httpx async keeps the pipeline non-blocking.

Key feature we exploit: `keyterms_prompt`. This is the custom-dictionary
hook the old SenseVoice path could never have (sherpa-onnx hotwords only
work on transducer models under modified_beam_search; SenseVoice is CTC).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import List, Optional

import httpx
import numpy as np

from .base import TranscriptionResult, WordInfo, float_to_wav_bytes

BASE_URL = "https://api.assemblyai.com"


def _http2_available() -> bool:
    """httpx raises ImportError at client construction if http2=True but the
    optional `h2` package is missing. Detect instead of crashing -- HTTP/1.1
    works fine, it just costs a little more connection overhead."""
    try:
        import h2  # noqa: F401
        return True
    except ImportError:
        return False

# Models that accept keyterms_prompt. universal-2 / "best" use the older,
# deprecated word_boost instead and will 400 on keyterms_prompt.
KEYTERM_MODELS = {"universal-3-5-pro", "universal-3-pro", "slam-1"}

# Terminal states returned by the polling endpoint.
_DONE = "completed"
_ERROR = "error"


class AssemblyAIClient:
    """Cloud STT via AssemblyAI. Implements the TranscriptionEngine protocol."""

    name = "assemblyai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "universal-3-5-pro",
        language_code: str = "en",
        request_timeout: float = 30.0,
        poll_interval: float = 0.25,
        max_wait: float = 60.0,
        format_text: bool = True,
        punctuate: bool = True,
        disfluencies: bool = False,
    ):
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY", "")
        self.model = model
        self.language_code = language_code
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait

        # format_text gives casing + ITN (numbers, dates, currency) server-side,
        # which removes a chunk of work from the LLM refinement stage.
        self.format_text = format_text
        self.punctuate = punctuate
        # disfluencies=False strips "um"/"uh" at the source. Cheaper and safer
        # than asking an LLM to remove them afterwards.
        self.disfluencies = disfluencies

        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def set_api_key(self, key: str) -> None:
        """Swap the key at runtime; forces a fresh client with new headers."""
        self.api_key = (key or "").strip()
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    base_url=BASE_URL,
                    timeout=httpx.Timeout(self.request_timeout, connect=10.0),
                    headers={"authorization": self.api_key},
                    http2=_http2_available(),
                    # Keep the TLS connection warm between dictations; a cold
                    # handshake costs ~200ms on every utterance otherwise.
                    limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300),
                )
            return self._client

    def preload(self) -> None:
        """No model to load, but we can warm the TLS connection."""
        return None

    async def warmup(self) -> None:
        """Open the HTTPS connection ahead of the first dictation."""
        if not self.is_configured:
            return
        try:
            client = await self._get_client()
            await client.get("/v2/transcript", params={"limit": 1}, timeout=5.0)
        except Exception:
            pass  # warmup is best-effort

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ── transcription ─────────────────────────────────────────────────────

    async def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        keyterms: Optional[List[str]] = None,
    ) -> TranscriptionResult:
        started = time.perf_counter()
        duration = len(samples) / float(sample_rate) if sample_rate else 0.0

        if not self.is_configured:
            return TranscriptionResult.failure(
                "No AssemblyAI API key configured.", self.name, self.model
            )

        if samples is None or len(samples) == 0:
            return TranscriptionResult.failure("No audio to transcribe.", self.name, self.model)

        try:
            wav_bytes = float_to_wav_bytes(samples, sample_rate)
            client = await self._get_client()

            audio_url = await self._upload(client, wav_bytes)
            transcript_id = await self._submit(client, audio_url, keyterms)
            payload = await self._poll(client, transcript_id)

            return self._parse(payload, started, duration)

        except httpx.TimeoutException:
            return TranscriptionResult.failure(
                "AssemblyAI request timed out.", self.name, self.model
            )
        except httpx.HTTPStatusError as e:
            return TranscriptionResult.failure(
                f"AssemblyAI HTTP {e.response.status_code}: {_body_snippet(e.response)}",
                self.name,
                self.model,
            )
        except Exception as e:
            return TranscriptionResult.failure(
                f"AssemblyAI error: {type(e).__name__}: {e}", self.name, self.model
            )

    async def _upload(self, client: httpx.AsyncClient, wav_bytes: bytes) -> str:
        resp = await client.post(
            "/v2/upload",
            content=wav_bytes,
            headers={"content-type": "application/octet-stream"},
        )
        resp.raise_for_status()
        return resp.json()["upload_url"]

    async def _submit(
        self,
        client: httpx.AsyncClient,
        audio_url: str,
        keyterms: Optional[List[str]],
    ) -> str:
        body = {
            "audio_url": audio_url,
            "speech_model": self.model,
            "language_code": self.language_code,
            "punctuate": self.punctuate,
            "format_text": self.format_text,
            "disfluencies": self.disfluencies,
        }

        # Only send keyterms to models that accept them; universal-2 rejects it.
        if keyterms and self.model in KEYTERM_MODELS:
            body["keyterms_prompt"] = _clean_keyterms(keyterms)

        resp = await client.post("/v2/transcript", json=body)
        resp.raise_for_status()
        return resp.json()["id"]

    async def _poll(self, client: httpx.AsyncClient, transcript_id: str) -> dict:
        endpoint = f"/v2/transcript/{transcript_id}"
        deadline = time.monotonic() + self.max_wait
        delay = self.poll_interval

        while True:
            resp = await client.get(endpoint)
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status")
            if status == _DONE:
                return data
            if status == _ERROR:
                raise RuntimeError(data.get("error", "transcription failed"))

            if time.monotonic() > deadline:
                raise TimeoutError(f"Transcript {transcript_id} not ready after {self.max_wait}s")

            await asyncio.sleep(delay)
            # Gentle backoff: dictation clips finish fast, so poll tightly at
            # first, then ease off to avoid hammering the API on slow jobs.
            delay = min(delay * 1.3, 1.0)

    def _parse(self, data: dict, started: float, duration: float) -> TranscriptionResult:
        text = (data.get("text") or "").strip()

        words = [
            WordInfo(
                text=w.get("text", ""),
                confidence=float(w.get("confidence", 1.0)),
                start=int(w.get("start", 0)),
                end=int(w.get("end", 0)),
            )
            for w in (data.get("words") or [])
        ]

        confidence = data.get("confidence")
        if confidence is None:
            confidence = sum(w.confidence for w in words) / len(words) if words else 1.0

        return TranscriptionResult(
            text=text,
            confidence=float(confidence),
            words=words,
            language=data.get("language_code") or self.language_code,
            engine=self.name,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            audio_duration_s=duration,
            ok=True,  # empty text here means genuine silence, not an error
        )

    def get_info(self) -> dict:
        return {
            "engine": "AssemblyAI",
            "model": self.model,
            "language": self.language_code,
            "configured": self.is_configured,
            "keyterms_supported": self.model in KEYTERM_MODELS,
        }


# ── helpers ───────────────────────────────────────────────────────────────


def _clean_keyterms(terms: List[str], limit: int = 1000, max_words: int = 6) -> List[str]:
    """AssemblyAI caps phrases at 6 words and the list at ~1000 terms."""
    out, seen = [], set()
    for t in terms:
        t = (t or "").strip()
        if not t or len(t.split()) > max_words:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _body_snippet(resp: httpx.Response, limit: int = 200) -> str:
    try:
        data = resp.json()
        return str(data.get("error") or data)[:limit]
    except Exception:
        return resp.text[:limit]
