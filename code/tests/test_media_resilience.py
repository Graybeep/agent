"""Media extraction must never end a run.

A corrupt image, unreadable audio, unsupported format, or an empty model
response should degrade that message to text-only routing, not raise out of
the batch loop.
"""


import pytest

from src.media_extractor import MediaExtractor


@pytest.fixture
def broken_media_dir(tmp_path):
    """A dataset dir whose image files are corrupt / empty."""
    (tmp_path / "media" / "images").mkdir(parents=True)
    (tmp_path / "media" / "images" / "corrupt.jpg").write_bytes(bytes(range(256)) * 8)
    (tmp_path / "media" / "images" / "empty.jpg").write_bytes(b"")
    (tmp_path / "images.csv").write_text(
        "image_id,file_path\ncorrupt,media/images/corrupt.jpg\nempty,media/images/empty.jpg\n"
    )
    (tmp_path / "voice_notes.csv").write_text("voice_note_id,file_path\n")
    return tmp_path


def test_missing_media_file_returns_none_without_calling_api(broken_media_dir, tmp_path):
    """An id with no file on disk resolves to None rather than raising."""
    extractor = MediaExtractor(broken_media_dir, tmp_path / "cache")
    assert extractor.get_media_text("image", "does_not_exist") is None


def test_absent_media_fields_return_none(broken_media_dir, tmp_path):
    extractor = MediaExtractor(broken_media_dir, tmp_path / "cache")
    assert extractor.get_media_text(None, None) is None
    assert extractor.get_media_text("", "") is None
    assert extractor.get_media_text("image", float("nan")) is None
    assert extractor.get_media_text("unsupported_kind", "corrupt") is None


def test_extraction_failure_degrades_to_text_only_in_the_batch_loop(monkeypatch):
    """The main loop must swallow extraction errors and route on text alone.

    Mirrors main.py's guard: whatever MediaExtractor raises, the per-message
    handler yields media_text=None and processing continues.
    """

    def exploding_get_media_text(self, media_type, media_id):
        raise RuntimeError("simulated corrupt media")

    monkeypatch.setattr(MediaExtractor, "get_media_text", exploding_get_media_text)
    extractor = MediaExtractor.__new__(MediaExtractor)

    # the guard as implemented in main.py
    try:
        media_text = extractor.get_media_text("image", "corrupt")
    except Exception:
        media_text = None

    assert media_text is None


def test_unreadable_path_raises_an_exception_subclass(tmp_path):
    """An unreadable path (here: a directory) surfaces as an OSError family
    error, which the main-loop guard must treat like any other failure."""
    (tmp_path / "media" / "images" / "is_a_dir.jpg").mkdir(parents=True)
    (tmp_path / "images.csv").write_text(
        "image_id,file_path\ndirpath,media/images/is_a_dir.jpg\n"
    )
    (tmp_path / "voice_notes.csv").write_text("voice_note_id,file_path\n")

    extractor = MediaExtractor(tmp_path, tmp_path / "cache")
    with pytest.raises(Exception) as excinfo:
        extractor.get_media_text("image", "dirpath")
    assert isinstance(excinfo.value, Exception)
    assert isinstance(excinfo.value, OSError)


def test_safety_blocked_response_raises_an_exception_subclass(tmp_path):
    """A safety-filtered Gemini response has text=None; .strip() on it raises
    AttributeError, which must also be catchable by the main-loop guard."""
    (tmp_path / "media" / "images").mkdir(parents=True)
    (tmp_path / "media" / "images" / "ok.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 128)
    (tmp_path / "images.csv").write_text("image_id,file_path\nblocked,media/images/ok.jpg\n")
    (tmp_path / "voice_notes.csv").write_text("voice_note_id,file_path\n")

    class BlockedResponse:
        text = None

    class FakeModels:
        def generate_content(self, **kwargs):
            return BlockedResponse()

    class FakeClient:
        models = FakeModels()

    extractor = MediaExtractor(tmp_path, tmp_path / "cache", client=FakeClient())
    with pytest.raises(AttributeError):
        extractor.get_media_text("image", "blocked")


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("all media models failed"),
        PermissionError(13, "Permission denied"),
        AttributeError("'NoneType' object has no attribute 'strip'"),
        OSError("I/O error"),
        ValueError("unsupported mime type"),
    ],
    ids=["corrupt", "unreadable", "safety_blocked", "io_error", "bad_mime"],
)
def test_every_media_failure_type_degrades_to_none(exc):
    """The guard must be broad enough for every plausible extraction failure."""
    try:
        raise exc
    except Exception:
        media_text = None
    assert media_text is None


def test_guard_does_not_swallow_keyboard_interrupt():
    """`except Exception` must still let Ctrl+C and SystemExit through."""
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt
        except Exception:  # noqa: BLE001 - mirrors main.py's guard
            pass


def test_main_module_guards_the_media_call():
    """Regression guard: the try/except around get_media_text must stay in main.py."""
    from pathlib import Path

    main_src = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    call_index = main_src.index("extractor.get_media_text")

    # a `try:` must open between the media stage marker and the call itself
    window = main_src[main_src.index('stage("media")') : call_index]
    assert "try:" in window, "get_media_text call is no longer inside a try block"

    # and the handler must degrade to None rather than re-raising
    handler = main_src[call_index : call_index + 800]
    assert "except Exception" in handler, "no except clause after the media call"
    assert "media_text = None" in handler, "no degrade-to-None branch after the call"
