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



query = "how many voting method used in multi agent debate system?"
#
# # 搜索全库
results = search(query, sentence_transformer_ef, collection_name="pure_algorithm", topk=3)
#
# # # 或者限定某篇文章
# # results = search(query, article_id="1", topk=3)
# #
# # # 或按标题过滤
# # results = search(query, title="A Milvus-based Retrieval-Augmented Generation Framework", topk=3)
#
for r in results:
    print(r)