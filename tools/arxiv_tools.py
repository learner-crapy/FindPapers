
import requests
import feedparser
import argparse
import json
import datetime


def extract_paper_info(entry):
    paper_info = {
        "title": entry.title,
        "authors": [{"name": author.name, "affiliation": getattr(author, 'affiliation', 'N/A')} for author in entry.authors],
        "published": entry.published,
        "summary": entry.summary,
        "link": entry.link,
        "categories": [category.term for category in entry.tags],
        "doi": getattr(entry, 'arxiv_doi', None),
        "journal_ref": getattr(entry, 'arxiv_journal_ref', None),
        "comments": getattr(entry, 'arxiv_comment', None),
    }
    return paper_info


def fetch_arxiv_data(start_date, end_date, search_query, max_results=5, use_date_filter=False):
    base_url = "http://export.arxiv.org/api/query?"

    if use_date_filter:
        date_query = f"submittedDate:[{start_date} TO {end_date}]"
        full_query = f"{search_query} AND {date_query}"
    else:
        full_query = search_query

    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    response = requests.get(base_url, params=params)
    feed = feedparser.parse(response.content)
    return [extract_paper_info(entry) for entry in feed.entries]





