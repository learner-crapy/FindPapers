from pathlib import Path

from tools.rag_tools import *
from pymilvus import MilvusClient
from glob import glob

import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"   # macOS GPU
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

from pymilvus import model
sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    device=DEVICE
)
client = MilvusClient(host="localhost", port=1953)

text_lines = []

ensure_collection(client, sentence_transformer_ef.dim, "pure_algorithm")
# 遍历所有 Markdown 文件
for i, file_path in enumerate(glob("milvus_docs/pure_algorithm/*.md", recursive=True)):
    file_path = Path(file_path)
    title = file_path.stem
    article_id = title[:12]   # 用文件名作为唯一ID

    with open(file_path, "r", encoding="utf-8") as file:
        file_text = file.read()

    # 拆分文本（简单按标题分块）
    text_lines = [chunk.strip() for chunk in file_text.split("# ") if chunk.strip()]

    # 插入到 Milvus
    insert_chunks(
        client=client,
        article_id=article_id,
        title=title,
        chunks=text_lines,
        embedding_fn=sentence_transformer_ef,
        collection_name="pure_algorithm"
    )

print("✅ 所有 Markdown 文件已插入到 Milvus。")
delete_paper(client, "1", "pure_algorithm")


# query = "What is the purpose of Milvus in a RAG system?"
#
# # 搜索全库
# results = search(query, sentence_transformer_ef, collection_name="pure_algorithm", topk=3)
#
# # # 或者限定某篇文章
# # results = search(query, article_id="1", topk=3)
# #
# # # 或按标题过滤
# # results = search(query, title="A Milvus-based Retrieval-Augmented Generation Framework", topk=3)
#
# for r in results:
#     print(r)