from __future__ import annotations

import json
import os
import uuid


class Chunker:
    # Dùng cho bài viết crawl thật (dài 2000-5000 ký tự).
    # Với medical_knowledge.jsonl (mỗi dòng ~150 ký tự) thì embedder đọc thẳng,
    # không cần qua đây.

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str) -> list[dict]:
        text = text.strip()
        if not text:
            return []

        chunks = []
        step   = self.chunk_size - self.chunk_overlap  # bước nhảy, tạo ra phần overlap
        i      = 0

        while i < len(text):
            chunk = text[i : i + self.chunk_size].strip()
            if chunk:
                chunks.append({"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk)), "text": chunk, "source": source})
            i += step

        return chunks

    def chunk_file(self, filepath: str) -> list[dict]:
        source = os.path.basename(filepath)

        if filepath.endswith(".txt"):
            with open(filepath, encoding="utf-8") as f:
                return self.chunk_text(f.read(), source)

        if filepath.endswith(".json"):
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            # Thử lần lượt các tên field phổ biến
            text   = data.get("text") or data.get("content") or data.get("body") or ""
            source = data.get("source", source)
            return self.chunk_text(text, source)

        return []

    def chunk_directory(self, dir_path: str) -> list[dict]:
        if not os.path.isdir(dir_path):
            return []

        all_chunks = []
        for filename in sorted(os.listdir(dir_path)):
            if filename.endswith((".txt", ".json")):
                all_chunks.extend(self.chunk_file(os.path.join(dir_path, filename)))
        return all_chunks

