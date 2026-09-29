"""Chunking: loaders per extension, metadata, fallbacks."""

from app.core.vectorstore import chunk_file


def test_txt_chunking_and_metadata():
    chunks, n_docs = chunk_file(
        b"Emberfall is a village. " * 60, "note.txt", source_name="note.txt"
    )
    assert n_docs == 1
    assert len(chunks) >= 2  # exceeds default 512-char chunk size
    assert all(
        c.metadata["source"] == "note.txt" and c.metadata["filename"] == "note.txt"
        for c in chunks
    )


def test_md_uses_text_loader():
    chunks, n_docs = chunk_file(
        b"# Emberfall\n\nThe Gilded Boar stands at the square.\n" * 20,
        "lore.md",
        source_name="lore.md",
    )
    assert n_docs == 1
    assert chunks
    assert "Gilded Boar" in chunks[0].page_content


def test_unknown_extension_falls_back_to_text():
    chunks, _ = chunk_file(b"plain content", "data.csv", source_name="data.csv")
    assert len(chunks) == 1
    assert chunks[0].page_content.strip() == "plain content"


def test_chunks_overlap():
    text = ("word%d " % i for i in range(400))
    chunks, _ = chunk_file(
        "".join(text).encode(), "big.txt", source_name="big.txt"
    )
    assert len(chunks) >= 2
    # overlap: tail of one chunk appears in the head of the next
    assert chunks[0].page_content[-20:] in chunks[1].page_content
