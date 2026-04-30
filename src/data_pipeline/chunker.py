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
                chunks.append({"id": str(uuid.uuid4()), "text": chunk, "source": source})
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


if __name__ == "__main__":
    chunker = Chunker(chunk_size=500, chunk_overlap=100)

    sample = (
        "Người bệnh đái tháo đường type 2 cần kiểm soát chặt chẽ lượng carbohydrate nạp vào. "
        "Thực phẩm có chỉ số đường huyết thấp như gạo lứt, khoai lang, yến mạch và các loại đậu "
        "là lựa chọn tốt hơn so với cơm trắng, bánh mì trắng hay đường tinh luyện. "
        "Rau xanh không tinh bột như bông cải xanh, rau muống, dưa leo nên chiếm phần lớn bữa ăn. "
        "Protein nạc từ ức gà, cá, đậu phụ giúp no lâu mà không làm tăng đường huyết đột ngột."
    )

    chunks = chunker.chunk_text(sample, source="test_article.txt")
    print(f"Số chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"\n[{i+1}] ({len(c['text'])} ký tự) {c['text'][:80]}...")
