import argparse
import asyncio
import datetime
import json
import os
import sys
from ollama import chat
from ollama import ChatResponse
from tqdm import tqdm

from AutoLog import get_logger
from resource import USER_PROMPT, IfHighlyAbout
from tools.db_operations import connect_db, get_existing_paper, update_existing_paper, insert_paper, update_paper_info
from tools.llm_operators import classify_if_match, extract_key_information
from tools.page_extract import get_web_page_content, get_web_content
logger = get_logger("find papers")

# 假设 arxiv_tools.py 在 ~/Downloads/n8n 目录下
project_root = os.path.expanduser("/Users/dazelu/DAZELU/Agent/FindPapers")

# 加到环境变量 PYTHONPATH
os.environ["PYTHONPATH"] = project_root

# 同时加到 sys.path，确保立即生效
if project_root not in sys.path:
    sys.path.append(project_root)

from tools.arxiv_tools import fetch_arxiv_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch papers from arXiv")
    parser.add_argument("--query", type=str, required=True, help="Search query (e.g., agent, LLM, reinforcement learning)")
    parser.add_argument("--days", type=int, default=1, help="How many past days to search")
    parser.add_argument("--max_results", type=int, default=100, help="Max number of results to fetch")
    parser.add_argument("--web_type", type=str, choices=['pdf', 'web'], default='pdf', help='you can select from pdf and web')

    args = parser.parse_args()

    # Date range
    end_date = datetime.datetime.now(datetime.UTC)
    start_date = end_date - datetime.timedelta(days=args.days)

    # Format as YYYYMMDDHHMM
    start_str = start_date.strftime("%Y%m%d%H%M")
    end_str = end_date.strftime("%Y%m%d%H%M")

    papers = fetch_arxiv_data(start_str, end_str, args.query, args.max_results)
    with connect_db() as conn:
        with conn.cursor() as cur:
            for i, paper in enumerate(tqdm(papers,desc="processing papers", disable=True)):
                # print(i, result, paper['published'], paper['link'], '-->', paper['title'])
                link = paper['link']
                existing = get_existing_paper(cur, link)
                if existing:
                    existing_queries, existing_ifmatch, existing_keyinfo = existing
                    if args.query in existing_queries:
                        logger.info(f"[{i}] Same query & link already exists → skip")
                        continue

                    # New query: run IfHighlyAbout
                    new_if_match = classify_if_match(paper)
                    logger.info(f"[{i}] Existing link, new query='{args.query}', if_match={new_if_match}")

                    key_info_text = ""
                    if new_if_match:
                        paper_html_link = link.replace("abs", "html") if args.web_type == "web" else link.replace("abs", "pdf")
                        try:
                            paper_text = asyncio.run(get_web_content(paper_html_link, args.web_type))
                        except Exception as e:
                            logger.error(f"Fetch HTML failed for {link}: {e}")
                            paper_text = ""
                        if paper_text.strip():
                            key_info_text = extract_key_information(paper_text)
                            update_existing_paper(cur, link, existing_queries, existing_ifmatch, args.query,
                                                  new_if_match, key_info_text)
                        else:
                            update_existing_paper(cur, link, existing_queries, existing_ifmatch, args.query,
                                                  new_if_match)
                    else:
                        update_existing_paper(cur, link, existing_queries, existing_ifmatch, args.query, new_if_match)

                    conn.commit()
                    continue

                # -------------------------------
                # Case 2: New paper (not exists)
                # -------------------------------
                if_match = classify_if_match(paper)
                insert_paper(cur, paper, args.query, if_match)
                conn.commit()

                if not if_match:
                    logger.warn(f"[{i}] Skipped: no True match found for {paper['title']}")
                    continue

                paper_html_link = link.replace("abs", "html") if args.web_type == "web" else link.replace("abs", "pdf")
                try:
                    paper_text = asyncio.run(get_web_content(paper_html_link, args.web_type))
                except Exception as e:
                    logger.error(f"Fetch HTML failed for {link}: {e}")
                    paper_text = ""

                key_info = ""
                if paper_text.strip():
                    key_info = extract_key_information(paper_text)

                update_paper_info(cur, link, key_info, paper_text)
                conn.commit()
                logger.info(f"[{i}] Added new paper with key_info: {paper['title']}")





