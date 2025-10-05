import datetime
import json

import psycopg

from resource import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT


def create_papers_table():
    # Database connection settings
    DB_NAME = "dazelu"
    DB_USER = ""
    DB_PASSWORD = ""
    DB_HOST = "localhost"
    DB_PORT = 5432

    with psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                link TEXT PRIMARY KEY,         -- use arXiv link as the unique ID
                title TEXT NOT NULL,
                authors JSONB,                 -- list of {name, affiliation}
                published TIMESTAMP,
                summary TEXT,
                categories TEXT[],             -- e.g. ['cs.AI', 'cs.MA']
                doi TEXT,
                journal_ref TEXT,
                comments TEXT,
                query TEXT[],                    -- the search query
                if_match BOOLEAN[],            -- list of bools
                key_information TEXT,          -- extracted reasoning
                paper_text TEXT                -- full text or long description
            )
            """)
            conn.commit()
            print("Table 'papers' created successfully (link as primary key).")


def connect_db():
    return psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )


def get_existing_paper(cur, link: str):
    """Return existing paper record if found."""
    cur.execute('SELECT query, if_match, key_information FROM papers WHERE link = %s', (link,)) # a tuple, must be % as the placeholder
    return cur.fetchone() # get one line, feachall(), get all


def insert_paper(cur, paper: dict, query_text: str, if_match: bool):
    """Insert a new paper record."""
    cur.execute("""
        INSERT INTO papers (
            link, title, authors, published, summary, categories,
            doi, journal_ref, comments, query, if_match
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (link) DO NOTHING
    """, (
        paper['link'],
        paper['title'],
        json.dumps(paper.get('authors', [])),
        paper.get('published'),
        paper.get('summary'),
        paper.get('categories', []),
        paper.get('doi'),
        paper.get('journal_ref'),
        paper.get('comments'),
        [query_text],
        [if_match]
    ))


def update_existing_paper(cur, link: str, existing_queries, existing_if_match, new_query, new_if_match, new_keyinfo=None):
    """Append new query / if_match / keyinfo if applicable."""
    # Merge queries
    if new_query not in existing_queries:
        existing_queries.append(new_query)
    # Append if_match
    existing_if_match.append(new_if_match)
    # Merge key_info  如果 key_information 是 NULL，就用空字符串 '' 代替。|| means 拼接
    if new_keyinfo:
        cur.execute("""
            UPDATE papers
            SET query = %s,
                if_match = %s,
                key_information = COALESCE(key_information, '') || %s
            WHERE link = %s
        """, (
            existing_queries,
            existing_if_match,
            "\n\n---\n\n" + new_keyinfo,
            link
        ))
    else:
        cur.execute("""
            UPDATE papers
            SET query = %s,
                if_match = %s
            WHERE link = %s
        """, (
            existing_queries,
            existing_if_match,
            link
        ))


def update_paper_info(cur, link: str, key_info: str, paper_text: str):
    cur.execute("""
        UPDATE papers
        SET key_information = %s,
            paper_text = %s
        WHERE link = %s
    """, (key_info, paper_text, link))




if __name__ == '__main__':
    create_papers_table()