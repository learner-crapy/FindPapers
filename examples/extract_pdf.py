import asyncio
import os
import time

import psycopg
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy

from examples.download_pdf import sanitize_filename
from tools.page_extract import get_pdf_page_content

conn = psycopg.connect(
    dbname="dazelu",
    user="",
    password="",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# ---- 查询匹配论文 ----
query = """
SELECT link, title
FROM papers
WHERE TRUE = ANY (if_match)
  AND '(\"agent debate\" OR \"multi-agent debate\") AND (reasoning OR performance OR system) NOT (\"external knowledge\" OR \"retrieval\" OR \"knowledge base\" OR \"database\")' = ANY(query);
        """
cur.execute(query)
results = cur.fetchall()

print(f"🔍 Found {len(results)} matched papers.\n")

# ---- 下载目录 ----
save_dir = "../milvus_docs/pure_algorithm/"
os.makedirs(save_dir, exist_ok=True)

# ---- 遍历下载 ----
for link, title in results:
    time.sleep(2)
    pdf_link = link.replace("/abs/", "/pdf/")
    safe_title = sanitize_filename(title) + ".md"
    file_path = os.path.join(save_dir, safe_title)
    result = asyncio.run(get_pdf_page_content(pdf_link))
    if result:
        with open(file_path, "w") as f:
            f.write(result)


# ---- 关闭连接 ----
cur.close()
conn.close()





