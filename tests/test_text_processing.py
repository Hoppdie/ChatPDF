"""Tests for ChatPDF's document ingestion pipeline.

These cover the two pure functions that decide what the retriever ever gets
to see: PDF text extraction and chunking. If chunking silently returns an
empty list or drops the overlap, the RAG answers degrade with no error --
exactly the kind of regression a test suite should catch.
"""
import io

import pytest

from app import get_chunks, get_pdf_text


def _make_pdf(lines):
    """Build a minimal single-page PDF in memory containing `lines`."""
    reportlab = pytest.importorskip("reportlab", reason="reportlab needed to synthesise a PDF")
    from reportlab.pdfgen import canvas  # noqa: WPS433

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    y = 800
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 20
    pdf.save()
    buf.seek(0)
    return buf


class TestGetChunks:
    def test_splits_long_text_into_multiple_chunks(self):
        # 60 newline-separated blocks of 100 chars each = ~6000 chars,
        # comfortably more than the 1000-char chunk_size.
        raw = "\n".join(["x" * 100 for _ in range(60)])
        chunks = get_chunks(raw)

        assert len(chunks) > 1, "long input must split into several chunks"
        assert all(isinstance(c, str) for c in chunks)
        assert all(c.strip() for c in chunks), "no chunk should be blank"

    def test_short_text_stays_single_chunk(self):
        chunks = get_chunks("a short paragraph that fits comfortably")
        assert len(chunks) == 1
        assert "short paragraph" in chunks[0]

    def test_chunks_respect_size_limit(self):
        raw = "\n".join(["y" * 200 for _ in range(40)])
        chunks = get_chunks(raw)
        # CharacterTextSplitter cannot break an atomic separator-free run,
        # so allow a tolerance but catch runaway chunks.
        assert max(len(c) for c in chunks) <= 1200

    def test_content_is_preserved(self):
        raw = "\n".join([f"sentence number {i}" for i in range(200)])
        joined = " ".join(get_chunks(raw))
        assert "sentence number 0" in joined
        assert "sentence number 199" in joined

    def test_empty_input_yields_no_chunks(self):
        assert get_chunks("") == []


class TestGetPdfText:
    def test_extracts_text_from_a_real_pdf(self):
        pdf = _make_pdf(["Retrieval augmented generation", "second line of text"])
        text = get_pdf_text([pdf])

        assert "Retrieval augmented generation" in text
        assert "second line" in text

    def test_concatenates_multiple_documents(self):
        first = _make_pdf(["alpha document"])
        second = _make_pdf(["beta document"])

        text = get_pdf_text([first, second])

        assert "alpha document" in text
        assert "beta document" in text

    def test_no_documents_returns_empty_string(self):
        assert get_pdf_text([]) == ""


def test_extraction_feeds_chunking():
    """The two stages must compose: PDF -> text -> chunks."""
    pdf = _make_pdf([f"paragraph {i} with meaningful content" for i in range(30)])
    chunks = get_chunks(get_pdf_text([pdf]))

    assert len(chunks) >= 1
    assert any("paragraph" in c for c in chunks)
