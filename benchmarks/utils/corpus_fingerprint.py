import hashlib
from pathlib import Path

from app.utils.document_id import (
    extract_document_id,
)


def compute_corpus_sha256(
    documents_directory: Path,
) -> str:
    """
    Compute a deterministic SHA-256 fingerprint for a benchmark corpus.
    """

    corpus_hasher = hashlib.sha256()

    document_paths = sorted(documents_directory.glob("DOC-*.md"))

    for document_path in document_paths:
        text = document_path.read_text(
            encoding="utf-8",
        )

        document_id = extract_document_id(document_path)

        corpus_hasher.update(document_id.encode("utf-8"))
        corpus_hasher.update(b"\0")
        corpus_hasher.update(text.encode("utf-8"))
        corpus_hasher.update(b"\0")

    return corpus_hasher.hexdigest()
