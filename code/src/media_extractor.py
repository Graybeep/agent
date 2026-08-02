"""Stage 1: multimodal pre-extraction for image and voice-note messages.

Uses the Gemini API for both modalities: audio files from voice_notes.csv are
transcribed and images from images.csv get verbatim text extraction plus a
short description. Results are cached to JSON on disk so repeated runs never
re-process the same media file.
"""

from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from .config import CACHE_DIR as DEFAULT_CACHE_DIR
from .config import DATASET_DIR as DEFAULT_DATASET_DIR
from .config import GEMINI_MODELS as MEDIA_MODELS
from .config import RATE_LIMIT_BACKOFF_S

load_dotenv()

IMAGE_PROMPT = (
    "You are helping route WhatsApp messages. Look at this image sent in a chat. "
    "Extract ALL visible text verbatim (posters, screenshots, QR-code captions, prices, "
    "URLs, phone numbers). Then add a one-sentence description of what the image is. "
    "Respond as plain text in this format:\n"
    "TEXT: <all visible text, or 'none'>\n"
    "DESCRIPTION: <one sentence>"
)

AUDIO_PROMPT = (
    "Transcribe this WhatsApp voice note verbatim. If it mixes languages, transcribe "
    "what is said and keep it readable. Respond with only the transcript text."
)


class MediaExtractor:
    """Extracts and caches text content from image and audio media files."""

    def __init__(
        self,
        dataset_dir: str | Path = DEFAULT_DATASET_DIR,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        client: genai.Client | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client

        self.image_paths = self._load_media_index("images.csv", "image_id")
        self.audio_paths = self._load_media_index("voice_notes.csv", "voice_note_id")

        self._image_cache_file = self.cache_dir / "image_text.json"
        self._audio_cache_file = self.cache_dir / "audio_transcripts.json"
        self._image_cache = _load_json(self._image_cache_file)
        self._audio_cache = _load_json(self._audio_cache_file)

    @property
    def client(self) -> genai.Client:
        # Lazy so cache-only usage never requires an API key.
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def _load_media_index(self, filename: str, id_column: str) -> dict[str, Path]:
        df = pd.read_csv(self.dataset_dir / filename)
        return {row[id_column]: self.dataset_dir / row["file_path"] for _, row in df.iterrows()}

    def _extract_from_file(self, path: Path, prompt: str, default_mime: str) -> str:
        mime = mimetypes.guess_type(path.name)[0] or default_mime
        contents = [
            types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
            prompt,
        ]
        last_error: Exception | None = None
        for model in MEDIA_MODELS:
            for attempt in range(len(RATE_LIMIT_BACKOFF_S) + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(temperature=0),
                    )
                    return response.text.strip()
                except errors.APIError as exc:
                    last_error = exc
                    if exc.code in (429, 503) and attempt < len(RATE_LIMIT_BACKOFF_S):
                        time.sleep(RATE_LIMIT_BACKOFF_S[attempt])
                        continue
                    break  # retries exhausted or non-retryable; try the next model
        raise RuntimeError(f"All media models failed for {path.name}") from last_error

    # ---- images --------------------------------------------------------------

    def get_image_text(self, image_id: str) -> str | None:
        """Return extracted text+description for an image, using cache first."""
        if image_id in self._image_cache:
            return self._image_cache[image_id]

        path = self.image_paths.get(image_id)
        if path is None or not path.exists():
            return None

        extracted = self._extract_from_file(path, IMAGE_PROMPT, "image/jpeg")
        self._image_cache[image_id] = extracted
        _save_json(self._image_cache_file, self._image_cache)
        return extracted

    # ---- audio ---------------------------------------------------------------

    def get_audio_transcript(self, voice_note_id: str) -> str | None:
        """Return the transcript for a voice note, using cache first."""
        if voice_note_id in self._audio_cache:
            return self._audio_cache[voice_note_id]

        path = self.audio_paths.get(voice_note_id)
        if path is None or not path.exists():
            return None

        transcript = self._extract_from_file(path, AUDIO_PROMPT, "audio/mpeg")
        self._audio_cache[voice_note_id] = transcript
        _save_json(self._audio_cache_file, self._audio_cache)
        return transcript

    # ---- unified lookup used by the routing pipeline -------------------------

    def get_media_text(self, media_type: str | None, media_id: str | None) -> str | None:
        """Return extracted text for a message's media, or None if no media."""
        if not media_type or not media_id or pd.isna(media_type) or pd.isna(media_id):
            return None
        if media_type == "image":
            return self.get_image_text(media_id)
        if media_type == "voice":
            return self.get_audio_transcript(media_id)
        return None

    def prewarm(self) -> None:
        """Extract every media file referenced by messages.csv up front."""
        messages = pd.read_csv(self.dataset_dir / "messages.csv")
        media_rows = messages[messages["media_id"].notna()]
        for _, row in media_rows.iterrows():
            self.get_media_text(row["media_type"], row["media_id"])


def _load_json(path: Path) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_json(path: Path, data: dict[str, str]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
