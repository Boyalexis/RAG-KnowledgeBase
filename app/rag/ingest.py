
import hashlib
import re
from pathlib import Path
from urllib.parse import quote

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
DISHES_DIR = ROOT / "data" / "sources" / "HowToCook" / "dishes"
DB_DIR = ROOT / "data" / "chroma"

MODEL_NAME = "/mnt/workspace/models/bge-small-zh-v1.5"
COLLECTION_NAME = "howtocook_recipes"

CHUNK_SIZE = 480
CHUNK_OVERLAP = 60
BATCH_SIZE = 32


def split_markdown(content: str, title: str):
    """按二级标题拆分，过长的章节再进行滑动窗口切分。"""

    sections = re.split(
        r"(?=^##\s+)",
        content,
        flags=re.MULTILINE,
    )

    for section in sections:
        section = section.strip()
        if not section:
            continue

        match = re.search(r"^##\s+(.+)", section, re.MULTILINE)
        section_name = match.group(1).strip() if match else "简介"

        prefix = f"菜名：{title}\n章节：{section_name}\n"
        window = CHUNK_SIZE - len(prefix)

        start = 0

        while start < len(section):
            end = min(start + window, len(section))
            yield prefix + section[start:end]

            if end == len(section):
                break

            start = end - CHUNK_OVERLAP


def load_chunks():
    """读取 dishes 下的 Markdown 并生成带元数据的 Chunk。"""

    if not DISHES_DIR.is_dir():
        raise FileNotFoundError(
            f"菜谱目录不存在：{DISHES_DIR}\n请先克隆 HowToCook 仓库。"
        )

    records = []
    files = sorted(DISHES_DIR.rglob("*.md"))

    for file in files:
        content = file.read_text(encoding="utf-8-sig").strip()

        if not content:
            continue

        relative = file.relative_to(DISHES_DIR)
        title = file.stem
        category = relative.parts[0] if len(relative.parts) > 1 else "other"

        source_url = (
            "https://github.com/Anduin2017/HowToCook/blob/master/dishes/"
            + quote(relative.as_posix(), safe="/")
        )

        for index, chunk in enumerate(split_markdown(content, title)):
            record_id = hashlib.sha1(
                f"{relative.as_posix()}:{index}".encode("utf-8")
            ).hexdigest()

            records.append({
                "id": record_id,
                "text": chunk,
                "metadata": {
                    "title": title,
                    "category": category,
                    "source": relative.as_posix(),
                    "source_url": source_url,
                },
            })

    return len(files), records


def ingest():
    file_count, records = load_chunks()

    print(f"发现 Markdown 文件：{file_count}")
    print(f"生成 Chunk 数量：{len(records)}")

    if not records:
        raise RuntimeError("没有可入库的 Markdown 内容。")

    print("正在加载 BGE Embedding 模型...")
    model = SentenceTransformer(
        MODEL_NAME,
        local_files_only=True,
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        documents = [item["text"] for item in batch]

        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=documents,
            metadatas=[item["metadata"] for item in batch],
            embeddings=embeddings,
        )

        print(f"已处理 {min(start + BATCH_SIZE, len(records))}/{len(records)}")

    print(f"ChromaDB 已保存记录数：{collection.count()}")
    print(f"数据库位置：{DB_DIR}")


if __name__ == "__main__":
    ingest()
