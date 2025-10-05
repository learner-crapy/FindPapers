from tools.page_extract import download_pdf

import os
import re
import psycopg
import requests

def sanitize_filename(name: str) -> str:
    """移除文件名中不允许的字符"""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()

def download_pdf(url: str, file_path: str):
    """下载 PDF 文件"""
    try:
        print(f"⬇️  Downloading: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        print(f"✅ Saved to: {file_path}\n")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")

def main():
    # ---- 数据库连接 ----
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
    WHERE TRUE = ANY(if_match);
    """
    cur.execute(query)
    results = cur.fetchall()

    print(f"🔍 Found {len(results)} matched papers.\n")

    # ---- 下载目录 ----
    save_dir = "downloaded_papers"
    os.makedirs(save_dir, exist_ok=True)

    # ---- 遍历下载 ----
    for link, title in results:
        pdf_link = link.replace("/abs/", "/pdf/")
        safe_title = sanitize_filename(title) + ".pdf"
        file_path = os.path.join(save_dir, safe_title)
        download_pdf(pdf_link, file_path)

    # ---- 关闭连接 ----
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()