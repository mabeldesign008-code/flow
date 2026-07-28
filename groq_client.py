import os
import re
import time
import asyncio
import httpx
from collections import deque
from dotenv import load_dotenv

load_dotenv()


class GroqClient:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"
        self.articulate_mode = False

        self._client: httpx.AsyncClient | None = None
        self.context_window: deque[str] = deque(maxlen=5)

        # Rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = 0.5
        self.rate_limit_retry_count = 0
        self.max_retries = 3
        self.use_basic_cleanup = True

    # ── lifecycle ──────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                http2=True,
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    keepalive_expiry=120,
                ),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── public API ─────────────────────────────────────────────

    async def refine(self, text: str, allow_fallback: bool = True, cursor_context: str = "") -> str:
        if not text or not text.strip():
            return text or ""

        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)

        prompt = (
            self._prompt_articulate(cursor_context) if self.articulate_mode
            else self._prompt_standard(cursor_context)
        )

        # Prepend cursor context to the user message if available
        user_message = text
        if cursor_context:
            user_message = f"[CONTEXT: {cursor_context}]\n\n{text}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.0,
            "max_tokens": max(len(text.split()) * 4, 256),
        }

        try:
            client = await self._get_client()
            self.last_request_time = time.time()
            resp = await client.post(self.base_url, json=payload)

            if resp.status_code == 429:
                return await self._handle_rate_limit(text, resp, allow_fallback)

            if resp.status_code != 200:
                print(f"[Groq] HTTP {resp.status_code}")
                return self._basic_cleanup(text) if allow_fallback else text

            self.rate_limit_retry_count = 0
            raw_result = resp.json()["choices"][0]["message"]["content"].strip()
            result = self._sanitize(raw_result, text)

            self.context_window.append(result)
            return result

        except httpx.TimeoutException:
            print("[Groq] Timeout")
            return self._basic_cleanup(text) if allow_fallback else text
        except Exception as e:
            print(f"[Groq] Error: {e}")
            return self._basic_cleanup(text) if allow_fallback else text

    # ── 🔒 ENGLISH-ONLY prompts ───────────────────────────────

    def _build_context(self) -> str:
        if not self.context_window:
            return "None."
        return "\n".join(
            f"  [{i+1}] {t}" for i, t in enumerate(self.context_window)
        )

    def _prompt_standard(self, cursor_context: str = "") -> str:
        ctx = self._build_context()
        context_section = ""
        if cursor_context:
            context_section = f"\nCursor Context (what the user is currently working on):\n{cursor_context}\n"
        
        return (
            "You are an English speech-to-text error corrector. "             # 🔒 ENGLISH-ONLY
            "You receive a raw English transcription and output a corrected version.\n\n"
            "Rules:\n"
            "1. Fix misheard words using surrounding context "
            "(e.g. 'cuban artists' → 'Kubernetes' when discussing containers).\n"
            "2. Fix grammar ONLY when it is clearly a transcription error, "
            "not the speaker's natural style.\n"
            "3. Preserve meaning, tone, and intent exactly.\n"
            "4. Remove pure filler (um, uh, you know, like) when they add nothing.\n"
            "5. Fix punctuation and capitalization.\n"
            "6. NEVER flip negations — carefully preserve 'can' vs 'can't', etc.\n"
            "7. Convert spoken numbers when natural (twenty percent → 20%).\n"
            "8. Use the recent context below AND the cursor context to resolve ambiguous references "
            "(e.g. 'that one', 'the same thing', pronouns).\n"
            "9. The input is ALWAYS English. If any word looks like another language, "   # 🔒
            "it is a misheard English word — correct it using context.\n"                 # 🔒
            "10. Output must be English only.\n"                                        # 🔒
            "11. If cursor context shows the user is working in a specific domain "
            "(code, email, document), use that to improve technical term accuracy.\n\n"
            f"Recent context:\n{ctx}\n"
            f"{context_section}\n"
            "Output ONLY the corrected text. "
            "No quotes, no explanation, no commentary."
        )

    def _prompt_articulate(self, cursor_context: str = "") -> str:
        ctx = self._build_context()
        context_section = ""
        if cursor_context:
            context_section = f"\nCursor Context (what the user is currently working on):\n{cursor_context}\n"
        
        return (
            "You are an English speech articulation engine. "                 # 🔒 ENGLISH-ONLY
            "Transform raw, rambling English speech into clear, polished, professional text.\n\n"
            "Rules:\n"
            "1. Understand the speaker's intent; express it concisely and clearly.\n"
            "2. Remove redundancy and filler, but keep every distinct idea.\n"
            "3. Improve structure, flow, and vocabulary.\n"
            "4. Fix all transcription errors using context.\n"
            "5. If the input is a QUESTION → output a polished question. "
            "Do NOT answer it.\n"
            "6. If the input is a COMMAND → output a polished command. "
            "Do NOT execute it.\n"
            "7. Never add information absent from the original.\n"
            "8. Never drop important context or qualifiers.\n"
            "9. The input is ALWAYS English. Treat any non-English-looking word "         # 🔒
            "as a misheard English word and correct it from context.\n"                   # 🔒
            "10. Output must be English only.\n"                                        # 🔒
            "11. If cursor context shows the user is working in a specific domain "
            "(code, email, document), adapt vocabulary and style appropriately.\n\n"
            f"Recent context:\n{ctx}\n"
            f"{context_section}\n"
            "Output ONLY the refined text. "
            "No quotes, no explanation, no commentary."
        )

    # ── output sanitization ────────────────────────────────────

    _PREAMBLE_RE = re.compile(
        r"^(?:here(?:'s| is) the (?:corrected|refined|cleaned|fixed) text\s*[:.]?\s*)",
        re.IGNORECASE,
    )

    def _sanitize(self, result: str, original: str) -> str:
        if not result:
            return original

        result = self._PREAMBLE_RE.sub("", result).strip()

        if len(result) >= 2:
            if (result[0] == '"' and result[-1] == '"') or \
               (result[0] == "'" and result[-1] == "'"):
                result = result[1:-1].strip()

        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()

        return result or original

    # ── fallback ───────────────────────────────────────────────

    @staticmethod
    def _basic_cleanup(text: str) -> str:
        if not text:
            return text
        t = text.strip()
        if t:
            t = t[0].upper() + t[1:]
        if t and t[-1] not in ".!?":
            t += "."
        return t

    # ── rate-limit machinery ───────────────────────────────────

    async def _handle_rate_limit(self, text, resp, allow_fallback):
        self.rate_limit_retry_count += 1
        if self.rate_limit_retry_count < self.max_retries:
            wait = min(
                int(resp.headers.get("retry-after", 5)),
                2 ** self.rate_limit_retry_count * 2,
            )
            print(f"[Groq] Rate-limited. Retry {self.rate_limit_retry_count}/{self.max_retries} in {wait}s")
            await asyncio.sleep(wait)
            return await self.refine(text, allow_fallback)

        print("[Groq] Rate limit — max retries exhausted")
        return self._basic_cleanup(text) if allow_fallback else text

    # ── status helpers ─────────────────────────────────────────

    def get_rate_limit_status(self):
        return {
            "retry_count": self.rate_limit_retry_count,
            "max_retries": self.max_retries,
            "time_since_last": time.time() - self.last_request_time,
        }

    def reset_rate_limit_count(self):
        self.rate_limit_retry_count = 0