from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any, Optional
from pymilvus import model
# 假设这些常量已在你项目中定义
MILVUS_URI = "http://localhost:19530"
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"
NLIST = 128
import math

MAX_TEXT_LEN = 4000  # Milvus VARCHAR 限制（可与 ensure_collection 保持一致）
BATCH_SIZE = 100



def ensure_collection(client: MilvusClient, dim, collection_name: str):
    """
    创建用于 RAG 的 Collection，带文章标题字段。
    Schema:
      - id (主键)
      - article_id
      - title
      - text
      - embedding
    """

    if client.has_collection(collection_name):
        print(f"✅ Collection `{collection_name}` 已存在，复用。")
        return

    # 启用动态字段，方便后续扩展
    schema = client.create_schema(auto_id=False, enable_dynamic_fields=True)

    # 字段定义
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("article_id", DataType.VARCHAR, max_length=128)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

    # 向量索引配置
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type=INDEX_TYPE,
        metric_type=METRIC_TYPE,
        params={"nlist": NLIST} if "IVF" in INDEX_TYPE else {}
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    print(f"✅ Collection `{collection_name}` 已创建完成！")




def split_long_text(text: str, max_len: int = 4096) -> list[str]:
    """
    递归对半切分超长文本，确保每段长度 <= max_len。
    优先在句号或空格处分割，最后强制截断。
    """
    text = text.strip()

    # ✅ 如果文本已经足够短，则直接返回（并保证不超限）
    if len(text) <= max_len:
        return [text]

    mid = len(text) // 2

    # 优先按句号或空格切分
    split_pos = text.rfind('.', 0, mid)
    if split_pos == -1:
        split_pos = text.rfind(' ', 0, mid)
    if split_pos == -1:
        split_pos = mid  # 实在找不到就硬切

    left = text[:split_pos + 1].strip()
    right = text[split_pos + 1:].strip()

    # ✅ 如果左/右仍然太长，递归切分
    left_parts = split_long_text(left, max_len) if len(left) > max_len else [left]
    right_parts = split_long_text(right, max_len) if len(right) > max_len else [right]

    # ✅ 拼接后统一「硬截断」以防 strip 导致越界
    result = []
    for chunk in (left_parts + right_parts):
        if len(chunk) > max_len:
            # 对超长 chunk 硬截断成多个等长片段
            for i in range(0, len(chunk), max_len):
                result.append(chunk[i:i + max_len])
        else:
            result.append(chunk)
    return result


def insert_chunks(
    client: MilvusClient,
    article_id: str,
    title: str,
    chunks: List[str],
    embedding_fn,
    collection_name: str,
):
    """
    将一篇文章的文本片段批量插入到 Milvus。
    自动处理超长文本、递归切分。
    每条记录包含：
      - id（可选）: 唯一键 (article_id_index)
      - article_id: 文章唯一标识
      - title: 文章标题
      - text: 内容（如超长自动切分）
      - embedding: 向量
    """

    if not chunks:
        print(f"⚠️ article_id={article_id} 没有可插入内容，跳过。")
        return

    # ⚙️ Step 1: 预处理所有 chunk（自动切分超长文本）
    safe_chunks = []
    for chunk in chunks:
        safe_chunks.extend(split_long_text(chunk, MAX_TEXT_LEN))

    print(f"🧩 文章 {title} 原 {len(chunks)} 段，经切分后 {len(safe_chunks)} 段。")

    # ⚙️ Step 2: 批量生成向量（SentenceTransformerEmbeddingFunction）
    embeddings = embedding_fn.encode_documents(safe_chunks)

    # ⚙️ Step 3: 组装插入行
    rows = []
    for i, (text, emb) in enumerate(zip(safe_chunks, embeddings)):
        safe_text = text[:MAX_TEXT_LEN]
        rows.append({
            "article_id": article_id,
            "title": title,
            "text": safe_text,
            "embedding": emb.tolist(),
        })

    # ⚙️ Step 4: 分批写入
    total_batches = math.ceil(len(rows) / BATCH_SIZE)
    for b in range(total_batches):
        batch = rows[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        client.insert(collection_name=collection_name, data=batch)
        print(f"   ✅ 批次 {b+1}/{total_batches} 已插入 {len(batch)} 条")

    print(f"✅ 成功插入 {len(rows)} 条记录到 `{collection_name}` (title={title})")

def delete_paper(client: MilvusClient, article_id: str, collection_name: str):
    """
    删除指定 article_id 的所有文档数据。
    兼容 pymilvus 2.6+，使用 filter 参数。
    """
    filter_expr = f"article_id == '{article_id}'"
    client.delete(
        collection_name=collection_name,
        filter=filter_expr  # ✅ 新版写法
    )
    print(f"🗑️ 已删除 article_id={article_id} 的记录。")

# -----------------------------
# 搜索函数
# -----------------------------
def search(
        query: str,
        embed_fn: Any,
        collection_name: str,
        article_id: Optional[str] = None,
        title: Optional[str] = None,
        topk: int = 5,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
    """
    使用 SentenceTransformerEmbeddingFunction 向量化查询，
    在 Milvus 中检索最相关的文本片段。
    """

    client = MilvusClient(uri=MILVUS_URI)

    # 1️⃣ 将 query 编码为向量
    qv = embed_fn.encode_queries([query])[0]

    # 2️⃣ 构建过滤表达式
    expr_parts = []
    if article_id:
        expr_parts.append(f"article_id == '{article_id}'")
    if title:
        expr_parts.append(f"title == '{title}'")
    expr = " and ".join(expr_parts) if expr_parts else None

    # 3️⃣ 检索
    raw_res = client.search(
        collection_name=collection_name,
        data=[qv],
        anns_field="embedding",
        limit=topk,
        search_params={"metric_type": "L2", "params": {"nprobe": 10}},
        filter=expr,  # ✅ 新版参数
        output_fields=output_fields or ["id", "article_id", "title", "text"]
    )

    # 4️⃣ 整理结果
    results = []
    for hits in raw_res:
        for hit in hits:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "distance": hit.distance,
                "query": query,
                **hit.entity
            })

    return results


if __name__ == "__main__":
    pass