import argparse
import asyncio
import datetime
import json
import os
import sys
import psycopg
from tqdm import tqdm
from ollama import chat, ChatResponse

from tools.page_extract import get_web_page_content
from resource import USER_PROMPT, IfHighlyAbout, KEY_INFO_PROMPT
from tools.arxiv_tools import fetch_arxiv_data


# -------------------------------
# PostgreSQL connection settings
# -------------------------------
DB_NAME = "dazelu"
DB_USER = "postgres"
DB_PASSWORD = "yourpassword"
DB_HOST = "localhost"
DB_PORT = 5432


# -------------------------------
# Database Helper Functions
# -------------------------------



















# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and store arXiv papers")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--max_results", type=int, default=500)
    args = parser.parse_args()

    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=args.days)
    start_str = start_date.strftime("%Y%m%d%H%M")
    end_str = end_date.strftime("%Y%m%d%H%M")

    papers = fetch_arxiv_data(start_str, end_str, args.query, args.max_results)

    processed_count = 0
    skipped_count = 0
    updated_count = 0
    new_count = 0

    with connect_db() as conn:
        with conn.cursor() as cur:
            for i, paper in enumerate(tqdm(papers, desc="processing papers")):
                link = paper["link"]
                existing = get_existing_paper(cur, link)

                # -------------------------------
                # Case 1: Existing record found
                # -------------------------------
                if existing:
                    existing_queries, existing_ifmatch, existing_keyinfo = existing
                    if args.query in existing_queries:
                        print(f"⏭️ [{i}] Same query & link already exists → skip")
                        skipped_count += 1
                        continue

                    # New query: run IfHighlyAbout
                    new_if_match = classify_if_match(paper)
                    print(f"🔄 [{i}] Existing link, new query='{args.query}', if_match={new_if_match}")

                    key_info_text = ""
                    if new_if_match:
                        paper_html_link = link.replace("abs", "html")
                        try:
                            paper_text = asyncio.run(get_web_page_content(paper_html_link))
                        except Exception as e:
                            print(f"⚠️ Fetch HTML failed for {link}: {e}")
                            paper_text = ""
                        if paper_text.strip():
                            key_info_text = extract_key_information(paper_text)
                            update_existing_paper(cur, link, existing_queries, existing_ifmatch, args.query, new_if_match, key_info_text)
                        else:
                            update_existing_paper(cur, link, existing_queries, existing_ifmatch, args.query, new_if_match)
                    else:
                        update_existing_paper(cur, link, existing_queries, existing_ifmatch, args.query, new_if_match)

                    conn.commit()
                    updated_count += 1
                    continue

                # -------------------------------
                # Case 2: New paper (not exists)
                # -------------------------------
                if_match = classify_if_match(paper)
                insert_paper(cur, paper, args.query, if_match)
                conn.commit()
                new_count += 1

                if not if_match:
                    print(f"⏭️ [{i}] Skipped: no True match found for {paper['title']}")
                    continue

                paper_html_link = link.replace("abs", "html")
                try:
                    paper_text = asyncio.run(get_web_page_content(paper_html_link))
                except Exception as e:
                    print(f"⚠️ Fetch HTML failed for {link}: {e}")
                    paper_text = ""

                key_info = ""
                if paper_text.strip():
                    key_info = extract_key_information(paper_text)

                update_paper_info(cur, link, key_info, paper_text)
                conn.commit()
                processed_count += 1
                print(f"✅ [{i}] Added new paper with key_info: {paper['title']}")

    print(f"\n🎉 Done. New papers: {new_count}, Updated: {updated_count}, Processed: {processed_count}, Skipped: {skipped_count}")