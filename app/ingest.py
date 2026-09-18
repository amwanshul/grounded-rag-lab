from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    start: int
    end: int


def normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip()


def chunk_text(document_id: str, title: str, text: str, size: int = 420, overlap: int = 80) -> list[Chunk]:
    text = normalize(text)
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-chunk-{index}",
                    document_id=document_id,
                    title=title,
                    text=piece,
                    start=start,
                    end=end,
                )
            )
            index += 1

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def load_demo_corpus(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.txt")):
        chunks.extend(
            chunk_text(
                document_id=path.stem,
                title=path.stem.replace("-", " ").title(),
                text=path.read_text(encoding="utf-8"),
            )
        )
    return chunks
