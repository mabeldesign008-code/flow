"""
Engine router: picks the transcription engine and falls back cleanly.

Policy
------
  cloud_first (default) : AssemblyAI, fall back to local on failure
  cloud_only            : AssemblyAI, no fallback (fail loudly)
  local_only            : offline, never touches the network (privacy mode)

The fallback only fires on genuine *failures* (ok=False). An empty-but-ok
result means the user really was silent, and retrying locally would just
waste a second and risk inventing text.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import List, Optional

import numpy as np

from .assemblyai_client import AssemblyAIClient
from .base import TranscriptionResult
from .local_client import LocalSherpaClient


class EnginePolicy(str, Enum):
    CLOUD_FIRST = "cloud_first"
    CLOUD_ONLY = "cloud_only"
    LOCAL_ONLY = "local_only"


class EngineRouter:
    def __init__(
        self,
        cloud: Optional[AssemblyAIClient] = None,
        local: Optional[LocalSherpaClient] = None,
        policy: EnginePolicy = EnginePolicy.CLOUD_FIRST,
        on_event=None,
    ):
        self.cloud = cloud if cloud is not None else AssemblyAIClient()
        self.local = local
        self.policy = policy
        self._on_event = on_event or (lambda msg: None)

        self.last_result: Optional[TranscriptionResult] = None
        self.consecutive_cloud_failures = 0

    # ── lifecycle ─────────────────────────────────────────────────────────

    def preload(self) -> None:
        """Warm whichever engines we might need."""
        if self.policy is not EnginePolicy.CLOUD_ONLY and self.local is not None:
            self.local.preload()

    async def warmup(self) -> None:
        if self.policy is not EnginePolicy.LOCAL_ONLY and self.cloud.is_configured:
            await self.cloud.warmup()

    async def close(self) -> None:
        await asyncio.gather(
            self.cloud.close(),
            self.local.close() if self.local else asyncio.sleep(0),
            return_exceptions=True,
        )

    # ── transcription ─────────────────────────────────────────────────────

    async def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        keyterms: Optional[List[str]] = None,
    ) -> TranscriptionResult:
        result = await self._route(samples, sample_rate, keyterms)
        self.last_result = result
        return result

    async def _route(self, samples, sample_rate, keyterms) -> TranscriptionResult:
        if self.policy is EnginePolicy.LOCAL_ONLY:
            if self.local is None:
                return TranscriptionResult.failure("Local engine not configured.")
            return await self.local.transcribe(samples, sample_rate, keyterms)

        if not self.cloud.is_configured:
            if self.policy is EnginePolicy.CLOUD_ONLY:
                return TranscriptionResult.failure(
                    "No AssemblyAI API key. Add one in Settings."
                )
            if self.local is not None:
                self._on_event("No cloud key -- using local engine.")
                return await self.local.transcribe(samples, sample_rate, keyterms)
            return TranscriptionResult.failure("No transcription engine available.")

        cloud_result = await self.cloud.transcribe(samples, sample_rate, keyterms)

        if cloud_result.ok:
            self.consecutive_cloud_failures = 0
            return cloud_result

        self.consecutive_cloud_failures += 1

        if self.policy is EnginePolicy.CLOUD_ONLY or self.local is None:
            return cloud_result

        self._on_event(f"Cloud failed ({cloud_result.error}) -- falling back to local.")
        local_result = await self.local.transcribe(samples, sample_rate, keyterms)

        if not local_result.ok:
            # Report the cloud error: it is the one the user can act on
            # (bad key, no credit, no network) -- not the local symptom.
            return TranscriptionResult.failure(
                f"Cloud: {cloud_result.error} | Local: {local_result.error}"
            )

        local_result.error = f"(fallback after cloud error: {cloud_result.error})"
        return local_result

    # ── config ────────────────────────────────────────────────────────────

    def set_policy(self, policy: EnginePolicy) -> None:
        self.policy = policy
        if policy is not EnginePolicy.CLOUD_ONLY and self.local is not None:
            self.local.preload()

    def get_info(self) -> dict:
        return {
            "policy": self.policy.value,
            "cloud": self.cloud.get_info(),
            "local": self.local.get_info() if self.local else None,
            "consecutive_cloud_failures": self.consecutive_cloud_failures,
        }
